# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
""" This module is responsible to run the analyzer commands. """

import re
import os
import os.path
import tempfile
import functools
import subprocess
import logging
from libscanbuild.compilation import classify_source, compiler_language
from libscanbuild.clang import get_version, get_arguments
from libscanbuild.shell import decode

__all__ = ['run']

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
          'error_type', 'error_output', 'exit_code'])
def report_failure(opts):
    """ Create report when analyzer failed.

    The major report is the preprocessor output. The output filename generated
    randomly. The compiler output also captured into '.stderr.txt' file.
    And some more execution context also saved into '.info.txt' file. """

    def extension(opts):
        """ Generate preprocessor file extension. """

        mapping = {'objective-c++': '.mii', 'objective-c': '.mi', 'c++': '.ii'}
        return mapping.get(opts['language'], '.i')

    def destination(opts):
        """ Creates failures directory if not exits yet. """

        name = os.path.join(opts['output_dir'], 'failures')
        if not os.path.isdir(name):
            os.makedirs(name)
        return name

    error = opts['error_type']
    (handle, name) = tempfile.mkstemp(suffix=extension(opts),
                                      prefix='clang_' + error + '_',
                                      dir=destination(opts))
    os.close(handle)
    cwd = opts['directory']
    cmd = get_arguments([opts['clang'], '-fsyntax-only', '-E'] +
                        opts['flags'] + [opts['file'], '-o', name], cwd)
    logging.debug('exec command in %s: %s', cwd, ' '.join(cmd))
    subprocess.call(cmd, cwd=cwd)
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
    # return with the previous step exit code and output
    return {
        'error_output': opts['error_output'],
        'exit_code': opts['exit_code']
    }


@require(['clang', 'directory', 'flags', 'direct_args', 'file', 'output_dir',
          'output_format'])
def run_analyzer(opts, continuation=report_failure):
    """ It assembles the analysis command line and executes it. Capture the
    output of the analysis and returns with it. If failure reports are
    requested, it calls the continuation to generate it. """

    def output():
        """ Creates output file name for reports. """
        if opts['output_format'] in {'plist', 'plist-html'}:
            (handle, name) = tempfile.mkstemp(prefix='report-',
                                              suffix='.plist',
                                              dir=opts['output_dir'])
            os.close(handle)
            return name
        return opts['output_dir']

    cwd = opts['directory']
    cmd = get_arguments([opts['clang'], '--analyze'] + opts['direct_args'] +
                        opts['flags'] + [opts['file'], '-o', output()],
                        cwd)
    logging.debug('exec command in %s: %s', cwd, ' '.join(cmd))
    child = subprocess.Popen(cmd,
                             cwd=cwd,
                             universal_newlines=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    output = child.stdout.readlines()
    child.stdout.close()
    # do report details if it were asked
    child.wait()
    if opts.get('output_failures', False) and child.returncode:
        error_type = 'crash' if child.returncode & 127 else 'other_error'
        opts.update({
            'error_type': error_type,
            'error_output': output,
            'exit_code': child.returncode
        })
        return continuation(opts)
    # return the output for logging and exit code for testing
    return {'error_output': output, 'exit_code': child.returncode}


@require(['flags', 'force_debug'])
def filter_debug_flags(opts, continuation=run_analyzer):
    """ Filter out nondebug macros when requested. """

    if opts.pop('force_debug'):
        # lazy implementation just append an undefine macro at the end
        opts.update({'flags': opts['flags'] + ['-UNDEBUG']})

    return continuation(opts)


@require(['file', 'directory'])
def set_file_path_relative(opts, continuation=filter_debug_flags):
    """ Set source file path to relative to the working directory.

    The only purpose of this function is to pass the SATestBuild.py tests. """

    opts.update({'file': os.path.relpath(opts['file'], opts['directory'])})

    return continuation(opts)


@require(['language', 'compiler', 'file', 'flags'])
def language_check(opts, continuation=set_file_path_relative):
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
