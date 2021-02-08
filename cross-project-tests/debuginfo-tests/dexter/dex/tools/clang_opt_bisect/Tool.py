# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Clang opt-bisect tool."""

from collections import defaultdict
import os
import csv
import re
import pickle

from dex.builder import run_external_build_script
from dex.command.ParseCommand import get_command_infos
from dex.debugger.Debuggers import run_debugger_subprocess
from dex.debugger.DebuggerControllers.DefaultController import DefaultController
from dex.dextIR.DextIR import DextIR
from dex.heuristic import Heuristic
from dex.tools import TestToolBase
from dex.utils.Exceptions import DebuggerException, Error
from dex.utils.Exceptions import BuildScriptException, HeuristicException
from dex.utils.PrettyOutputBase import Stream
from dex.utils.ReturnCode import ReturnCode


class BisectPass(object):
    def __init__(self, no, description, description_no_loc):
        self.no = no
        self.description = description
        self.description_no_loc = description_no_loc

        self.penalty = 0
        self.differences = []


class Tool(TestToolBase):
    """Use the LLVM "-opt-bisect-limit=<n>" flag to get information on the
    contribution of each LLVM pass to the overall DExTer score when using
    clang.

    Clang is run multiple times, with an increasing value of n, measuring the
    debugging experience at each value.
    """

    _re_running_pass = re.compile(
        r'^BISECT\: running pass \((\d+)\) (.+?)( \(.+\))?$')

    def __init__(self, *args, **kwargs):
        super(Tool, self).__init__(*args, **kwargs)
        self._all_bisect_pass_summary = defaultdict(list)

    @property
    def name(self):
        return 'DExTer clang opt bisect'

    def _get_bisect_limits(self):
        options = self.context.options

        max_limit = 999999
        limits = [max_limit for _ in options.source_files]
        all_passes = [
            l for l in self._clang_opt_bisect_build(limits)[1].splitlines()
            if l.startswith('BISECT: running pass (')
        ]

        results = []
        for i, pass_ in enumerate(all_passes[1:]):
            if pass_.startswith('BISECT: running pass (1)'):
                results.append(all_passes[i])
        results.append(all_passes[-1])

        assert len(results) == len(
            options.source_files), (results, options.source_files)

        limits = [
            int(Tool._re_running_pass.match(r).group(1)) for r in results
        ]

        return limits

    def handle_options(self, defaults):
        options = self.context.options
        if "clang" not in options.builder.lower():
            raise Error("--builder %s is not supported by the clang-opt-bisect tool - only 'clang' is "
                        "supported " % options.builder)
        super(Tool, self).handle_options(defaults)

    def _init_debugger_controller(self):
        step_collection = DextIR(
            executable_path=self.context.options.executable,
            source_paths=self.context.options.source_files,
            dexter_version=self.context.version)

        step_collection.commands, new_source_files = get_command_infos(
            self.context.options.source_files, self.context.options.source_root_dir)
        self.context.options.source_files.extend(list(new_source_files))

        debugger_controller = DefaultController(self.context, step_collection)
        return debugger_controller

    def _run_test(self, test_name):  # noqa
        options = self.context.options

        per_pass_score = []
        current_bisect_pass_summary = defaultdict(list)

        max_limits = self._get_bisect_limits()
        overall_limit = sum(max_limits)
        prev_score = 1.0
        prev_steps_str = None

        for current_limit in range(overall_limit + 1):
            # Take the overall limit number and split it across buckets for
            # each source file.
            limit_remaining = current_limit
            file_limits = [0] * len(max_limits)
            for i, max_limit in enumerate(max_limits):
                if limit_remaining < max_limit:
                    file_limits[i] += limit_remaining
                    break
                else:
                    file_limits[i] = max_limit
                    limit_remaining -= file_limits[i]

            f = [l for l in file_limits if l]
            current_file_index = len(f) - 1 if f else 0

            _, err, builderIR = self._clang_opt_bisect_build(file_limits)
            err_lines = err.splitlines()
            # Find the last line that specified a running pass.
            for l in err_lines[::-1]:
                match = Tool._re_running_pass.match(l)
                if match:
                    pass_info = match.groups()
                    break
            else:
                pass_info = (0, None, None)

            try:
                debugger_controller =self._init_debugger_controller()
                debugger_controller = run_debugger_subprocess(
                    debugger_controller, self.context.working_directory.path)
                steps = debugger_controller.step_collection
            except DebuggerException:
                steps =  DextIR(
                    executable_path=self.context.options.executable,
                    source_paths=self.context.options.source_files,
                    dexter_version=self.context.version)

            steps.builder = builderIR

            try:
                heuristic = Heuristic(self.context, steps)
            except HeuristicException as e:
                raise Error(e)

            score_difference = heuristic.score - prev_score
            prev_score = heuristic.score

            isnan = heuristic.score != heuristic.score
            if isnan or score_difference < 0:
                color1 = 'r'
                color2 = 'r'
            elif score_difference > 0:
                color1 = 'g'
                color2 = 'g'
            else:
                color1 = 'y'
                color2 = 'd'

            summary = '<{}>running pass {}/{} on "{}"'.format(
                color2, pass_info[0], max_limits[current_file_index],
                test_name)
            if len(options.source_files) > 1:
                summary += ' [{}/{}]'.format(current_limit, overall_limit)

            pass_text = ''.join(p for p in pass_info[1:] if p)
            summary += ': {} <{}>{:+.4f}</> <{}>{}</></>\n'.format(
                heuristic.summary_string, color1, score_difference, color2,
                pass_text)

            self.context.o.auto(summary)

            heuristic_verbose_output = heuristic.verbose_output

            if options.verbose:
                self.context.o.auto(heuristic_verbose_output)

            steps_str = str(steps)
            steps_changed = steps_str != prev_steps_str
            prev_steps_str = steps_str

            # If this is the first pass, or something has changed, write a text
            # file containing verbose information on the current status.
            if current_limit == 0 or score_difference or steps_changed:
                file_name = '-'.join(
                    str(s) for s in [
                        'status', test_name, '{{:0>{}}}'.format(
                            len(str(overall_limit))).format(current_limit),
                        '{:.4f}'.format(heuristic.score).replace(
                            '.', '_'), pass_info[1]
                    ] if s is not None)

                file_name = ''.join(
                    c for c in file_name
                    if c.isalnum() or c in '()-_./ ').strip().replace(
                    ' ', '_').replace('/', '_')

                output_text_path = os.path.join(options.results_directory,
                                                '{}.txt'.format(file_name))
                with open(output_text_path, 'w') as fp:
                    self.context.o.auto(summary + '\n', stream=Stream(fp))
                    self.context.o.auto(str(steps) + '\n', stream=Stream(fp))
                    self.context.o.auto(
                        heuristic_verbose_output + '\n', stream=Stream(fp))

                output_dextIR_path = os.path.join(options.results_directory,
                                                  '{}.dextIR'.format(file_name))
                with open(output_dextIR_path, 'wb') as fp:
                    pickle.dump(steps, fp, protocol=pickle.HIGHEST_PROTOCOL)

            per_pass_score.append((test_name, pass_text,
                                   heuristic.score))

            if pass_info[1]:
                self._all_bisect_pass_summary[pass_info[1]].append(
                    score_difference)

                current_bisect_pass_summary[pass_info[1]].append(
                    score_difference)

        per_pass_score_path = os.path.join(
            options.results_directory,
            '{}-per_pass_score.csv'.format(test_name))

        with open(per_pass_score_path, mode='w', newline='') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(['Source File', 'Pass', 'Score'])

            for path, pass_, score in per_pass_score:
                writer.writerow([path, pass_, score])
        self.context.o.blue('wrote "{}"\n'.format(per_pass_score_path))

        pass_summary_path = os.path.join(
            options.results_directory, '{}-pass-summary.csv'.format(test_name))

        self._write_pass_summary(pass_summary_path,
                                 current_bisect_pass_summary)

    def _handle_results(self) -> ReturnCode:
        options = self.context.options
        pass_summary_path = os.path.join(options.results_directory,
                                         'overall-pass-summary.csv')

        self._write_pass_summary(pass_summary_path,
                                 self._all_bisect_pass_summary)
        return ReturnCode.OK

    def _clang_opt_bisect_build(self, opt_bisect_limits):
        options = self.context.options
        compiler_options = [
            '{} -mllvm -opt-bisect-limit={}'.format(options.cflags,
                                                    opt_bisect_limit)
            for opt_bisect_limit in opt_bisect_limits
        ]
        linker_options = options.ldflags

        try:
            return run_external_build_script(
                self.context,
                source_files=options.source_files,
                compiler_options=compiler_options,
                linker_options=linker_options,
                script_path=self.build_script,
                executable_file=options.executable)
        except BuildScriptException as e:
            raise Error(e)

    def _write_pass_summary(self, path, pass_summary):
        # Get a list of tuples.
        pass_summary_list = list(pass_summary.items())

        for i, item in enumerate(pass_summary_list):
            # Add elems for the sum, min, and max of the values, as well as
            # 'interestingness' which is whether any of these values are
            # non-zero.
            pass_summary_list[i] += (sum(item[1]), min(item[1]), max(item[1]),
                                     any(item[1]))

            # Split the pass name into the basic name and kind.
            pass_summary_list[i] += tuple(item[0].rsplit(' on ', 1))

        # Sort the list by the following columns in order of precedence:
        #   - Is interesting (True first)
        #   - Sum (smallest first)
        #   - Number of times pass ran (largest first)
        #   - Kind (alphabetically)
        #   - Name (alphabetically)
        pass_summary_list.sort(
            key=lambda tup: (not tup[5], tup[2], -len(tup[1]), tup[7], tup[6]))

        with open(path, mode='w', newline='') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(
                ['Pass', 'Kind', 'Sum', 'Min', 'Max', 'Interesting'])

            for (_, vals, sum_, min_, max_, interesting, name,
                 kind) in pass_summary_list:
                writer.writerow([name, kind, sum_, min_, max_, interesting] +
                                vals)

        self.context.o.blue('wrote "{}"\n'.format(path))
