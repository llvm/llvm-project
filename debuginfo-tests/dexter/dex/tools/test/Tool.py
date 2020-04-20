# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test tool."""

import math
import os
import csv
import pickle
import shutil

from dex.builder import run_external_build_script
from dex.command.ParseCommand import get_command_infos
from dex.debugger.Debuggers import run_debugger_subprocess
from dex.debugger.DebuggerControllers.DefaultController import DefaultController
from dex.dextIR.DextIR import DextIR
from dex.heuristic import Heuristic
from dex.tools import TestToolBase
from dex.utils.Exceptions import DebuggerException
from dex.utils.Exceptions import BuildScriptException, HeuristicException
from dex.utils.PrettyOutputBase import Stream
from dex.utils.ReturnCode import ReturnCode
from dex.dextIR import BuilderIR


class TestCase(object):
    def __init__(self, context, name, heuristic, error):
        self.context = context
        self.name = name
        self.heuristic = heuristic
        self.error = error

    @property
    def penalty(self):
        try:
            return self.heuristic.penalty
        except AttributeError:
            return float('nan')

    @property
    def max_penalty(self):
        try:
            return self.heuristic.max_penalty
        except AttributeError:
            return float('nan')

    @property
    def score(self):
        try:
            return self.heuristic.score
        except AttributeError:
            return float('nan')

    def __str__(self):
        if self.error and self.context.options.verbose:
            verbose_error = str(self.error)
        else:
            verbose_error = ''

        if self.error:
            script_error = (' : {}'.format(
                self.error.script_error.splitlines()[0]) if getattr(
                    self.error, 'script_error', None) else '')

            error = ' [{}{}]'.format(
                str(self.error).splitlines()[0], script_error)
        else:
            error = ''

        try:
            summary = self.heuristic.summary_string
        except AttributeError:
            summary = '<r>nan/nan (nan)</>'
        return '{}: {}{}\n{}'.format(self.name, summary, error, verbose_error)


class Tool(TestToolBase):
    """Run the specified DExTer test(s) with the specified compiler and linker
    options and produce a dextIR file as well as printing out the debugging
    experience score calculated by the DExTer heuristic.
    """

    def __init__(self, *args, **kwargs):
        super(Tool, self).__init__(*args, **kwargs)
        self._test_cases = []

    @property
    def name(self):
        return 'DExTer test'

    def add_tool_arguments(self, parser, defaults):
        parser.add_argument('--fail-lt',
                            type=float,
                            default=0.0, # By default TEST always succeeds.
                            help='exit with status FAIL(2) if the test result'
                                ' is less than this value.',
                            metavar='<float>')
        parser.add_argument('--calculate-average',
                            action="store_true",
                            help='calculate the average score of every test run')
        super(Tool, self).add_tool_arguments(parser, defaults)

    def _build_test_case(self):
        """Build an executable from the test source with the given --builder
        script and flags (--cflags, --ldflags) in the working directory.
        Or, if the --binary option has been given, copy the executable provided
        into the working directory and rename it to match the --builder output.
        """

        options = self.context.options
        if options.binary:
            # Copy user's binary into the tmp working directory
            shutil.copy(options.binary, options.executable)
            builderIR = BuilderIR(
                name='binary',
                cflags=[options.binary],
                ldflags='')
        else:
            options = self.context.options
            compiler_options = [options.cflags for _ in options.source_files]
            linker_options = options.ldflags
            _, _, builderIR = run_external_build_script(
                self.context,
                script_path=self.build_script,
                source_files=options.source_files,
                compiler_options=compiler_options,
                linker_options=linker_options,
                executable_file=options.executable)
        return builderIR

    def _init_debugger_controller(self):
        step_collection = DextIR(
            executable_path=self.context.options.executable,
            source_paths=self.context.options.source_files,
            dexter_version=self.context.version)
        step_collection.commands = get_command_infos(
            self.context.options.source_files)
        debugger_controller = DefaultController(self.context, step_collection)
        return debugger_controller

    def _get_steps(self, builderIR):
        """Generate a list of debugger steps from a test case.
        """
        debugger_controller = self._init_debugger_controller()
        debugger_controller = run_debugger_subprocess(
            debugger_controller, self.context.working_directory.path)
        steps = debugger_controller.step_collection
        steps.builder = builderIR
        return steps

    def _get_results_basename(self, test_name):
        def splitall(x):
            while len(x) > 0:
              x, y = os.path.split(x)
              yield y
        all_components = reversed([x for x in splitall(test_name)])
        return '_'.join(all_components)

    def _get_results_path(self, test_name):
        """Returns the path to the test results directory for the test denoted
        by test_name.
        """
        return os.path.join(self.context.options.results_directory,
                            self._get_results_basename(test_name))

    def _get_results_text_path(self, test_name):
        """Returns path results .txt file for test denoted by test_name.
        """
        test_results_path = self._get_results_path(test_name)
        return '{}.txt'.format(test_results_path)

    def _get_results_pickle_path(self, test_name):
        """Returns path results .dextIR file for test denoted by test_name.
        """
        test_results_path = self._get_results_path(test_name)
        return '{}.dextIR'.format(test_results_path)

    def _record_steps(self, test_name, steps):
        """Write out the set of steps out to the test's .txt and .json
        results file.
        """
        output_text_path = self._get_results_text_path(test_name)
        with open(output_text_path, 'w') as fp:
            self.context.o.auto(str(steps), stream=Stream(fp))

        output_dextIR_path = self._get_results_pickle_path(test_name)
        with open(output_dextIR_path, 'wb') as fp:
            pickle.dump(steps, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def _record_score(self, test_name, heuristic):
        """Write out the test's heuristic score to the results .txt file.
        """
        output_text_path = self._get_results_text_path(test_name)
        with open(output_text_path, 'a') as fp:
            self.context.o.auto(heuristic.verbose_output, stream=Stream(fp))

    def _record_test_and_display(self, test_case):
        """Output test case to o stream and record test case internally for
        handling later.
        """
        self.context.o.auto(test_case)
        self._test_cases.append(test_case)

    def _record_failed_test(self, test_name, exception):
        """Instantiate a failed test case with failure exception and
        store internally.
        """
        test_case = TestCase(self.context, test_name, None, exception)
        self._record_test_and_display(test_case)

    def _record_successful_test(self, test_name, steps, heuristic):
        """Instantiate a successful test run, store test for handling later.
        Display verbose output for test case if required.
        """
        test_case = TestCase(self.context, test_name, heuristic, None)
        self._record_test_and_display(test_case)
        if self.context.options.verbose:
            self.context.o.auto('\n{}\n'.format(steps))
            self.context.o.auto(heuristic.verbose_output)

    def _run_test(self, test_name):
        """Attempt to run test files specified in options.source_files. Store
        result internally in self._test_cases.
        """
        try:
            builderIR = self._build_test_case()
            steps = self._get_steps(builderIR)
            self._record_steps(test_name, steps)
            heuristic_score = Heuristic(self.context, steps)
            self._record_score(test_name, heuristic_score)
        except (BuildScriptException, DebuggerException,
                HeuristicException) as e:
            self._record_failed_test(test_name, e)
            return

        self._record_successful_test(test_name, steps, heuristic_score)
        return

    def _handle_results(self) -> ReturnCode:
        return_code = ReturnCode.OK
        options = self.context.options

        if not options.verbose:
            self.context.o.auto('\n')

        if options.calculate_average:
            # Calculate and print the average score
            score_sum = 0.0
            num_tests = 0
            for test_case in self._test_cases:
                score = test_case.score
                if not test_case.error and not math.isnan(score):
                    score_sum += test_case.score
                    num_tests += 1

            if num_tests != 0:
                print("@avg: ({:.4f})".format(score_sum/num_tests))

        summary_path = os.path.join(options.results_directory, 'summary.csv')
        with open(summary_path, mode='w', newline='') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(['Test Case', 'Score', 'Error'])

            for test_case in self._test_cases:
                if (test_case.score < options.fail_lt or
                        test_case.error is not None):
                    return_code = ReturnCode.FAIL

                writer.writerow([
                    test_case.name, '{:.4f}'.format(test_case.score),
                    test_case.error
                ])

        return return_code
