# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
""" This module implements the 'scan-build' command API.

To run the static analyzer against a build is done in multiple steps:

 -- Intercept: capture the compilation command during the build,
 -- Analyze:   run the analyzer against the captured commands,
 -- Report:    create a cover report from the analyzer outputs.  """

import re
import os
import os.path
import json
import logging
import multiprocessing
import tempfile
import functools
import subprocess
import contextlib
import datetime

from libscanbuild import command_entry_point, compiler_wrapper, \
    wrapper_environment, run_build, run_command
from libscanbuild.arguments import parse_args_for_scan_build, \
    parse_args_for_analyze_build
from libscanbuild.intercept import capture
from libscanbuild.report import document
from libscanbuild.compilation import split_command, classify_source, \
    compiler_language
from libscanbuild.clang import get_version, get_arguments
from libscanbuild.shell import decode

__all__ = ['scan_build', 'analyze_build', 'analyze_compiler_wrapper']

COMPILER_WRAPPER_CC = 'analyze-cc'
COMPILER_WRAPPER_CXX = 'analyze-c++'


@command_entry_point
def scan_build():
    """ Entry point for scan-build command. """

    args = parse_args_for_scan_build()
    # will re-assign the report directory as new output
    with report_directory(args.output, args.keep_empty) as args.output:
        # Run against a build command. there are cases, when analyzer run
        # is not required. But we need to set up everything for the
        # wrappers, because 'configure' needs to capture the CC/CXX values
        # for the Makefile.
        if args.intercept_first:
            # Run build command with intercept module.
            exit_code = capture(args)
            # Run the analyzer against the captured commands.
            if need_analyzer(args.build):
                run_analyzer_parallel(args)
        else:
            # Run build command and analyzer with compiler wrappers.
            environment = setup_environment(args)
            exit_code = run_build(args.build, env=environment)
        # Cover report generation and bug counting.
        number_of_bugs = document(args)
        # Set exit status as it was requested.
        return number_of_bugs if args.status_bugs else exit_code


@command_entry_point
def analyze_build():
    """ Entry point for analyze-build command. """

    args = parse_args_for_analyze_build()
    # will re-assign the report directory as new output
    with report_directory(args.output, args.keep_empty) as args.output:
        # Run the analyzer against a compilation db.
        run_analyzer_parallel(args)
        # Cover report generation and bug counting.
        number_of_bugs = document(args)
        # Set exit status as it was requested.
        return number_of_bugs if args.status_bugs else 0


def need_analyzer(args):
    """ Check the intent of the build command.

    When static analyzer run against project configure step, it should be
    silent and no need to run the analyzer or generate report.

    To run `scan-build` against the configure step might be neccessary,
    when compiler wrappers are used. That's the moment when build setup
    check the compiler and capture the location for the build process. """

    return len(args) and not re.search('configure|autogen', args[0])


def run_analyzer_parallel(args):
    """ Runs the analyzer against the given compilation database. """

    def exclude(filename):
        """ Return true when any excluded directory prefix the filename. """
        return any(re.match(r'^' + directory, filename)
                   for directory in args.excludes)

    consts = {
        'clang': args.clang,
        'output_dir': args.output,
        'output_format': args.output_format,
        'output_failures': args.output_failures,
        'direct_args': analyzer_params(args),
        'force_debug': args.force_debug
    }

    logging.debug('run analyzer against compilation database')
    with open(args.cdb, 'r') as handle:
        generator = (dict(cmd, **consts)
                     for cmd in json.load(handle) if not exclude(cmd['file']))
        # when verbose output requested execute sequentially
        pool = multiprocessing.Pool(1 if args.verbose > 2 else None)
        for current in pool.imap_unordered(run, generator):
            if current is not None:
                # display error message from the static analyzer
                for line in current['error_output']:
                    logging.info(line.rstrip())
        pool.close()
        pool.join()


def setup_environment(args):
    """ Set up environment for build command to interpose compiler wrapper. """

    environment = dict(os.environ)
    environment.update(wrapper_environment(args))
    environment.update({
        'CC': COMPILER_WRAPPER_CC,
        'CXX': COMPILER_WRAPPER_CXX,
        'ANALYZE_BUILD_CLANG': args.clang if need_analyzer(args.build) else '',
        'ANALYZE_BUILD_REPORT_DIR': args.output,
        'ANALYZE_BUILD_REPORT_FORMAT': args.output_format,
        'ANALYZE_BUILD_REPORT_FAILURES': 'yes' if args.output_failures else '',
        'ANALYZE_BUILD_PARAMETERS': ' '.join(analyzer_params(args)),
        'ANALYZE_BUILD_FORCE_DEBUG': 'yes' if args.force_debug else ''
    })
    return environment


@command_entry_point
def analyze_compiler_wrapper():
    """ Entry point for `analyze-cc` and `analyze-c++` compiler wrappers. """

    return compiler_wrapper(analyze_compiler_wrapper_impl)


def analyze_compiler_wrapper_impl(result, execution):
    """ Implements analyzer compiler wrapper functionality. """

    # don't run analyzer when compilation fails. or when it's not requested.
    if result or not os.getenv('ANALYZE_BUILD_CLANG'):
        return

    # check is it a compilation?
    compilation = split_command(execution.cmd)
    if compilation is None:
        return
    # collect the needed parameters from environment, crash when missing
    parameters = {
        'clang': os.getenv('ANALYZE_BUILD_CLANG'),
        'output_dir': os.getenv('ANALYZE_BUILD_REPORT_DIR'),
        'output_format': os.getenv('ANALYZE_BUILD_REPORT_FORMAT'),
        'output_failures': os.getenv('ANALYZE_BUILD_REPORT_FAILURES'),
        'direct_args': os.getenv('ANALYZE_BUILD_PARAMETERS',
                                 '').split(' '),
        'force_debug': os.getenv('ANALYZE_BUILD_FORCE_DEBUG'),
        'directory': execution.cwd,
        'command': [execution.cmd[0], '-c'] + compilation.flags
    }
    # call static analyzer against the compilation
    for source in compilation.files:
        parameters.update({'file': source})
        logging.debug('analyzer parameters %s', parameters)
        current = run(parameters)
        # display error message from the static analyzer
        if current is not None:
            for line in current['error_output']:
                logging.info(line.rstrip())


@contextlib.contextmanager
def report_directory(hint, keep):
    """ Responsible for the report directory.

    hint -- could specify the parent directory of the output directory.
    keep -- a boolean value to keep or delete the empty report directory. """

    stamp_format = 'scan-build-%Y-%m-%d-%H-%M-%S-%f-'
    stamp = datetime.datetime.now().strftime(stamp_format)
    parent_dir = os.path.abspath(hint)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    name = tempfile.mkdtemp(prefix=stamp, dir=parent_dir)

    logging.info('Report directory created: %s', name)

    try:
        yield name
    finally:
        if os.listdir(name):
            msg = "Run 'scan-view %s' to examine bug reports."
            keep = True
        else:
            if keep:
                msg = "Report directory '%s' contains no report, but kept."
            else:
                msg = "Removing directory '%s' because it contains no report."
        logging.warning(msg, name)

        if not keep:
            os.rmdir(name)


def analyzer_params(args):
    """ A group of command line arguments can mapped to command
    line arguments of the analyzer. This method generates those. """

    def prefix_with(constant, pieces):
        """ From a sequence create another sequence where every second element
        is from the original sequence and the odd elements are the prefix.

        eg.: prefix_with(0, [1,2,3]) creates [0, 1, 0, 2, 0, 3] """

        return [elem for piece in pieces for elem in [constant, piece]]

    result = []

    if args.store_model:
        result.append('-analyzer-store={0}'.format(args.store_model))
    if args.constraints_model:
        result.append('-analyzer-constraints={0}'.format(
            args.constraints_model))
    if args.internal_stats:
        result.append('-analyzer-stats')
    if args.analyze_headers:
        result.append('-analyzer-opt-analyze-headers')
    if args.stats:
        result.append('-analyzer-checker=debug.Stats')
    if args.maxloop:
        result.extend(['-analyzer-max-loop', str(args.maxloop)])
    if args.output_format:
        result.append('-analyzer-output={0}'.format(args.output_format))
    if args.analyzer_config:
        result.append(args.analyzer_config)
    if args.verbose >= 4:
        result.append('-analyzer-display-progress')
    if args.plugins:
        result.extend(prefix_with('-load', args.plugins))
    if args.enable_checker:
        checkers = ','.join(args.enable_checker)
        result.extend(['-analyzer-checker', checkers])
    if args.disable_checker:
        checkers = ','.join(args.disable_checker)
        result.extend(['-analyzer-disable-checker', checkers])
    if os.getenv('UBIVIZ'):
        result.append('-analyzer-viz-egraph-ubigraph')

    return prefix_with('-Xclang', result)


def require(required):
    """ Decorator for checking the required values in state.

    It checks the required attributes in the passed state and stop when
    any of those is missing. """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for key in required:
                if key not in args[0]:
                    raise KeyError('{0} not passed to {1}'.format(
                        key, function.__name__))

            return function(*args, **kwargs)

        return wrapper

    return decorator


@require(['command',  # entry from compilation database
          'directory',  # entry from compilation database
          'file',  # entry from compilation database
          'clang',  # clang executable name (and path)
          'direct_args',  # arguments from command line
          'force_debug',  # kill non debug macros
          'output_dir',  # where generated report files shall go
          'output_format',  # it's 'plist' or 'html' or both
          'output_failures'])  # generate crash reports or not
def run(opts):
    """ Entry point to run (or not) static analyzer against a single entry
    of the compilation database.

    This complex task is decomposed into smaller methods which are calling
    each other in chain. If the analyzis is not possibe the given method
    just return and break the chain.

    The passed parameter is a python dictionary. Each method first check
    that the needed parameters received. (This is done by the 'require'
    decorator. It's like an 'assert' to check the contract between the
    caller and the called method.) """

    try:
        command = opts.pop('command')
        command = command if isinstance(command, list) else decode(command)
        logging.debug("Run analyzer against '%s'", command)
        opts.update(classify_parameters(command))

        return arch_check(opts)
    except Exception:
        logging.error("Problem occured during analyzis.", exc_info=1)
        return None


@require(['clang', 'directory', 'flags', 'file', 'output_dir', 'language',
          'error_output', 'exit_code'])
def report_failure(opts):
    """ Create report when analyzer failed.

    The major report is the preprocessor output. The output filename generated
    randomly. The compiler output also captured into '.stderr.txt' file.
    And some more execution context also saved into '.info.txt' file. """

    def extension():
        """ Generate preprocessor file extension. """

        mapping = {'objective-c++': '.mii', 'objective-c': '.mi', 'c++': '.ii'}
        return mapping.get(opts['language'], '.i')

    def destination():
        """ Creates failures directory if not exits yet. """

        failures_dir = os.path.join(opts['output_dir'], 'failures')
        if not os.path.isdir(failures_dir):
            os.makedirs(failures_dir)
        return failures_dir

    # Classify error type: when Clang terminated by a signal it's a 'Crash'.
    # (python subprocess Popen.returncode is negative when child terminated
    # by signal.) Everything else is 'Other Error'.
    error = 'crash' if opts['exit_code'] < 0 else 'other_error'
    # Create preprocessor output file name. (This is blindly following the
    # Perl implementation.)
    (handle, name) = tempfile.mkstemp(suffix=extension(),
                                      prefix='clang_' + error + '_',
                                      dir=destination())
    os.close(handle)
    # Execute Clang again, but run the syntax check only.
    cwd = opts['directory']
    cmd = get_arguments(
        [opts['clang'], '-fsyntax-only', '-E'
         ] + opts['flags'] + [opts['file'], '-o', name], cwd)
    run_command(cmd, cwd=cwd)
    # write general information about the crash
    with open(name + '.info.txt', 'w') as handle:
        handle.write(opts['file'] + os.linesep)
        handle.write(error.title().replace('_', ' ') + os.linesep)
        handle.write(' '.join(cmd) + os.linesep)
        handle.write(' '.join(os.uname()) + os.linesep)
        handle.write(get_version(opts['clang']))
        handle.close()
    # write the captured output too
    with open(name + '.stderr.txt', 'w') as handle:
        handle.writelines(opts['error_output'])
        handle.close()


@require(['clang', 'directory', 'flags', 'direct_args', 'file', 'output_dir',
          'output_format'])
def run_analyzer(opts, continuation=report_failure):
    """ It assembles the analysis command line and executes it. Capture the
    output of the analysis and returns with it. If failure reports are
    requested, it calls the continuation to generate it. """

    def target():
        """ Creates output file name for reports. """
        if opts['output_format'] in {'plist', 'plist-html'}:
            (handle, name) = tempfile.mkstemp(prefix='report-',
                                              suffix='.plist',
                                              dir=opts['output_dir'])
            os.close(handle)
            return name
        return opts['output_dir']

    try:
        cwd = opts['directory']
        cmd = get_arguments([opts['clang'], '--analyze'] +
                            opts['direct_args'] + opts['flags'] +
                            [opts['file'], '-o', target()],
                            cwd)
        output = run_command(cmd, cwd=cwd)
        return {'error_output': output, 'exit_code': 0}
    except subprocess.CalledProcessError as ex:
        result = {'error_output': ex.output, 'exit_code': ex.returncode}
        if opts.get('output_failures', False):
            opts.update(result)
            continuation(opts)
        return result


@require(['flags', 'force_debug'])
def filter_debug_flags(opts, continuation=run_analyzer):
    """ Filter out nondebug macros when requested. """

    if opts.pop('force_debug'):
        # lazy implementation just append an undefine macro at the end
        opts.update({'flags': opts['flags'] + ['-UNDEBUG']})

    return continuation(opts)


@require(['language', 'compiler', 'file', 'flags'])
def language_check(opts, continuation=filter_debug_flags):
    """ Find out the language from command line parameters or file name
    extension. The decision also influenced by the compiler invocation. """

    accepted = frozenset({
        'c', 'c++', 'objective-c', 'objective-c++', 'c-cpp-output',
        'c++-cpp-output', 'objective-c-cpp-output'
    })

    # language can be given as a parameter...
    language = opts.pop('language')
    compiler = opts.pop('compiler')
    # ... or find out from source file extension
    if language is None and compiler is not None:
        language = classify_source(opts['file'], compiler == 'c')

    if language is None:
        logging.debug('skip analysis, language not known')
        return None
    elif language not in accepted:
        logging.debug('skip analysis, language not supported')
        return None
    else:
        logging.debug('analysis, language: %s', language)
        opts.update({'language': language,
                     'flags': ['-x', language] + opts['flags']})
        return continuation(opts)


@require(['arch_list', 'flags'])
def arch_check(opts, continuation=language_check):
    """ Do run analyzer through one of the given architectures. """

    disabled = frozenset({'ppc', 'ppc64'})

    received_list = opts.pop('arch_list')
    if received_list:
        # filter out disabled architectures and -arch switches
        filtered_list = [a for a in received_list if a not in disabled]
        if filtered_list:
            # There should be only one arch given (or the same multiple
            # times). If there are multiple arch are given and are not
            # the same, those should not change the pre-processing step.
            # But that's the only pass we have before run the analyzer.
            current = filtered_list.pop()
            logging.debug('analysis, on arch: %s', current)

            opts.update({'flags': ['-arch', current] + opts['flags']})
            return continuation(opts)
        else:
            logging.debug('skip analysis, found not supported arch')
            return None
    else:
        logging.debug('analysis, on default arch')
        return continuation(opts)

# To have good results from static analyzer certain compiler options shall be
# omitted. The compiler flag filtering only affects the static analyzer run.
#
# Keys are the option name, value number of options to skip
IGNORED_FLAGS = {
    '-c': 0,  # compile option will be overwritten
    '-fsyntax-only': 0,  # static analyzer option will be overwritten
    '-o': 1,  # will set up own output file
    # flags below are inherited from the perl implementation.
    '-g': 0,
    '-save-temps': 0,
    '-install_name': 1,
    '-exported_symbols_list': 1,
    '-current_version': 1,
    '-compatibility_version': 1,
    '-init': 1,
    '-e': 1,
    '-seg1addr': 1,
    '-bundle_loader': 1,
    '-multiply_defined': 1,
    '-sectorder': 3,
    '--param': 1,
    '--serialize-diagnostics': 1
}


def classify_parameters(command):
    """ Prepare compiler flags (filters some and add others) and take out
    language (-x) and architecture (-arch) flags for future processing. """

    result = {
        'flags': [],  # the filtered compiler flags
        'arch_list': [],  # list of architecture flags
        'language': None,  # compilation language, None, if not specified
        'compiler': compiler_language(command)  # 'c' or 'c++'
    }

    # iterate on the compile options
    args = iter(command[1:])
    for arg in args:
        # take arch flags into a separate basket
        if arg == '-arch':
            result['arch_list'].append(next(args))
        # take language
        elif arg == '-x':
            result['language'] = next(args)
        # parameters which looks source file are not flags
        elif re.match(r'^[^-].+', arg) and classify_source(arg):
            pass
        # ignore some flags
        elif arg in IGNORED_FLAGS:
            count = IGNORED_FLAGS[arg]
            for _ in range(count):
                next(args)
        # we don't care about extra warnings, but we should suppress ones
        # that we don't want to see.
        elif re.match(r'^-W.+', arg) and not re.match(r'^-Wno-.+', arg):
            pass
        # and consider everything else as compilation flag.
        else:
            result['flags'].append(arg)

    return result
