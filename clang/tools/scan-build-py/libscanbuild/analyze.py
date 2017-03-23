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
import tempfile
import multiprocessing
import contextlib
import datetime
from libscanbuild import command_entry_point, compiler_wrapper, \
    wrapper_environment, run_build
from libscanbuild.arguments import parse_args_for_scan_build, \
    parse_args_for_analyze_build
from libscanbuild.runner import run
from libscanbuild.intercept import capture
from libscanbuild.report import document
from libscanbuild.compilation import split_command

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
                run_analyzer(args)
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
        run_analyzer(args)
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


def run_analyzer(args):
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
