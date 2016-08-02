# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
""" This module is responsible to run the analyzer commands. """

import os
import os.path
import tempfile
import functools
import subprocess
import logging
from libscanbuild.command import classify_parameters, Action, classify_source
from libscanbuild.clang import get_arguments, get_version
from libscanbuild.shell import decode

__all__ = ['run']


def require(required):
    """ Decorator for checking the required values in state.

    It checks the required attributes in the passed state and stop when
    any of those is missing. """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for key in required:
                if key not in args[0]:
                    raise KeyError(
                        '{0} not passed to {1}'.format(key, function.__name__))

            return function(*args, **kwargs)

        return wrapper

    return decorator


@require(['command', 'directory', 'file',  # an entry from compilation database
          'clang', 'direct_args',  # compiler name, and arguments from command
          'output_dir', 'output_format', 'output_failures'])
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
        logging.debug("Run analyzer against '%s'", command)
        opts.update(classify_parameters(decode(command)))

        return action_check(opts)
    except Exception:
        logging.error("Problem occured during analyzis.", exc_info=1)
        return None


@require(['report', 'directory', 'clang', 'output_dir', 'language', 'file',
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
    cmd = get_arguments([opts['clang']] + opts['report'] + ['-o', name], cwd)
    logging.debug('exec command in %s: %s', cwd, ' '.join(cmd))
    subprocess.call(cmd, cwd=cwd)

    with open(name + '.info.txt', 'w') as handle:
        handle.write(opts['file'] + os.linesep)
        handle.write(error.title().replace('_', ' ') + os.linesep)
        handle.write(' '.join(cmd) + os.linesep)
        handle.write(' '.join(os.uname()) + os.linesep)
        handle.write(get_version(cmd[0]))
        handle.close()

    with open(name + '.stderr.txt', 'w') as handle:
        handle.writelines(opts['error_output'])
        handle.close()

    return {
        'error_output': opts['error_output'],
        'exit_code': opts['exit_code']
    }


@require(['clang', 'analyze', 'directory', 'output'])
def run_analyzer(opts, continuation=report_failure):
    """ It assembles the analysis command line and executes it. Capture the
    output of the analysis and returns with it. If failure reports are
    requested, it calls the continuation to generate it. """

    cwd = opts['directory']
    cmd = get_arguments([opts['clang']] + opts['analyze'] + opts['output'],
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
    return {'error_output': output, 'exit_code': child.returncode}


@require(['output_dir'])
def set_analyzer_output(opts, continuation=run_analyzer):
    """ Create output file if was requested.

    This plays a role only if .plist files are requested. """

    if opts.get('output_format') in {'plist', 'plist-html'}:
        with tempfile.NamedTemporaryFile(prefix='report-',
                                         suffix='.plist',
                                         delete=False,
                                         dir=opts['output_dir']) as output:
            opts.update({'output': ['-o', output.name]})
            return continuation(opts)
    else:
        opts.update({'output': ['-o', opts['output_dir']]})
        return continuation(opts)


@require(['file', 'directory', 'clang', 'direct_args', 'language',
          'output_dir', 'output_format', 'output_failures'])
def create_commands(opts, continuation=set_analyzer_output):
    """ Create command to run analyzer or failure report generation.

    It generates commands (from compilation database entries) which contains
    enough information to run the analyzer (and the crash report generation
    if that was requested). """

    common = []
    if 'arch' in opts:
        common.extend(['-arch', opts.pop('arch')])
    common.extend(opts.pop('compile_options', []))
    common.extend(['-x', opts['language']])
    common.append(os.path.relpath(opts['file'], opts['directory']))

    opts.update({
        'analyze': ['--analyze'] + opts['direct_args'] + common,
        'report': ['-fsyntax-only', '-E'] + common
    })

    return continuation(opts)


@require(['file', 'c++'])
def language_check(opts, continuation=create_commands):
    """ Find out the language from command line parameters or file name
    extension. The decision also influenced by the compiler invocation. """

    accepteds = {
        'c', 'c++', 'objective-c', 'objective-c++', 'c-cpp-output',
        'c++-cpp-output', 'objective-c-cpp-output'
    }

    key = 'language'
    language = opts[key] if key in opts else \
        classify_source(opts['file'], opts['c++'])

    if language is None:
        logging.debug('skip analysis, language not known')
        return None
    elif language not in accepteds:
        logging.debug('skip analysis, language not supported')
        return None
    else:
        logging.debug('analysis, language: %s', language)
        opts.update({key: language})
        return continuation(opts)


@require([])
def arch_check(opts, continuation=language_check):
    """ Do run analyzer through one of the given architectures. """

    disableds = {'ppc', 'ppc64'}

    key = 'archs_seen'
    if key in opts:
        # filter out disabled architectures and -arch switches
        archs = [a for a in opts[key] if a not in disableds]

        if not archs:
            logging.debug('skip analysis, found not supported arch')
            return None
        else:
            # There should be only one arch given (or the same multiple
            # times). If there are multiple arch are given and are not
            # the same, those should not change the pre-processing step.
            # But that's the only pass we have before run the analyzer.
            arch = archs.pop()
            logging.debug('analysis, on arch: %s', arch)

            opts.update({'arch': arch})
            del opts[key]
            return continuation(opts)
    else:
        logging.debug('analysis, on default arch')
        return continuation(opts)


@require(['action'])
def action_check(opts, continuation=arch_check):
    """ Continue analysis only if it compilation or link. """

    if opts.pop('action') <= Action.Compile:
        return continuation(opts)
    else:
        logging.debug('skip analysis, not compilation nor link')
        return None
