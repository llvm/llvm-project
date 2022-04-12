# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Calculate a 'score' based on some dextIR.
Assign penalties based on different commands to decrease the score.
1.000 would be a perfect score.
0.000 is the worst theoretical score possible.
"""

from collections import defaultdict, namedtuple, Counter
import difflib
import os
from itertools import groupby
from dex.command.StepValueInfo import StepValueInfo
from dex.command.commands.DexExpectWatchBase import format_address


PenaltyCommand = namedtuple('PenaltyCommand', ['pen_dict', 'max_penalty'])
# 'meta' field used in different ways by different things
PenaltyInstance = namedtuple('PenaltyInstance', ['meta', 'the_penalty'])


def add_heuristic_tool_arguments(parser):
    parser.add_argument(
        '--penalty-variable-optimized',
        type=int,
        default=3,
        help='set the penalty multiplier for each'
        ' occurrence of a variable that was optimized'
        ' away',
        metavar='<int>')
    parser.add_argument(
        '--penalty-misordered-values',
        type=int,
        default=3,
        help='set the penalty multiplier for each'
        ' occurrence of a misordered value.',
        metavar='<int>')
    parser.add_argument(
        '--penalty-irretrievable',
        type=int,
        default=4,
        help='set the penalty multiplier for each'
        " occurrence of a variable that couldn't"
        ' be retrieved',
        metavar='<int>')
    parser.add_argument(
        '--penalty-not-evaluatable',
        type=int,
        default=5,
        help='set the penalty multiplier for each'
        " occurrence of a variable that couldn't"
        ' be evaluated',
        metavar='<int>')
    parser.add_argument(
        '--penalty-missing-values',
        type=int,
        default=6,
        help='set the penalty multiplier for each missing'
        ' value',
        metavar='<int>')
    parser.add_argument(
        '--penalty-incorrect-values',
        type=int,
        default=7,
        help='set the penalty multiplier for each'
        ' occurrence of an unexpected value.',
        metavar='<int>')
    parser.add_argument(
        '--penalty-unreachable',
        type=int,
        default=4,  # XXX XXX XXX selected by random
        help='set the penalty for each line stepped onto that should'
        ' have been unreachable.',
        metavar='<int>')
    parser.add_argument(
        '--penalty-misordered-steps',
        type=int,
        default=2,  # XXX XXX XXX selected by random
        help='set the penalty for differences in the order of steps'
        ' the program was expected to observe.',
        metavar='<int>')
    parser.add_argument(
        '--penalty-missing-step',
        type=int,
        default=4,  # XXX XXX XXX selected by random
        help='set the penalty for the program skipping over a step.',
        metavar='<int>')
    parser.add_argument(
        '--penalty-incorrect-program-state',
        type=int,
        default=4,  # XXX XXX XXX selected by random
        help='set the penalty for the program never entering an expected state'
        ' or entering an unexpected state.',
        metavar='<int>')


class PenaltyLineRanges:
    def __init__(self, first_step, penalty):
        self.ranges = [(first_step, first_step)]
        self.penalty = penalty

    def add_step(self, next_step, penalty):
        last_range = self.ranges[-1]
        last_step = last_range[1]
        if (next_step == last_step + 1):
            self.ranges[-1] = (last_range[0], next_step)
        else:
            self.ranges.append((next_step, next_step))
        self.penalty += penalty

    def __str__(self):
        range_to_str = lambda r: str(r[0]) if r[0] == r[1] else f'{r[0]}-{r[1]}'
        if self.ranges[0][0] == self.ranges[-1][1]:
            text = f'step {self.ranges[0][0]}'
        else:
            step_list = ', '.join([range_to_str(r) for r in self.ranges])
            text = f'steps [{step_list}]'
        if self.penalty:
            text += ' <r>[-{}]</>'.format(self.penalty)
        return text


class Heuristic(object):
    def __init__(self, context, steps):
        self.context = context
        self.penalties = {}
        self.address_resolutions = {}

        worst_penalty = max([
            self.penalty_variable_optimized, self.penalty_irretrievable,
            self.penalty_not_evaluatable, self.penalty_incorrect_values,
            self.penalty_missing_values, self.penalty_unreachable,
            self.penalty_missing_step, self.penalty_misordered_steps
        ])

        # Before evaluating scoring commands, evaluate address values.
        try:
            for command in steps.commands['DexDeclareAddress']:
                command.address_resolutions = self.address_resolutions
                command.eval(steps)
        except KeyError:
            pass

        # Get DexExpectWatchType results.
        try:
            for command in steps.commands['DexExpectWatchType']:
                command.eval(steps)
                maximum_possible_penalty = min(3, len(
                    command.values)) * worst_penalty
                name, p = self._calculate_expect_watch_penalties(
                    command, maximum_possible_penalty)
                name = name + ' ExpectType'
                self.penalties[name] = PenaltyCommand(p,
                                                      maximum_possible_penalty)
        except KeyError:
            pass

        # Get DexExpectWatchValue results.
        try:
            for command in steps.commands['DexExpectWatchValue']:
                command.address_resolutions = self.address_resolutions
                command.eval(steps)
                maximum_possible_penalty = min(3, len(
                    command.values)) * worst_penalty
                name, p = self._calculate_expect_watch_penalties(
                    command, maximum_possible_penalty)
                name = name + ' ExpectValue'
                self.penalties[name] = PenaltyCommand(p,
                                                      maximum_possible_penalty)
        except KeyError:
            pass

        try:
            penalties = defaultdict(list)
            maximum_possible_penalty_all = 0
            for expect_state in steps.commands['DexExpectProgramState']:
                success = expect_state.eval(steps)
                p = 0 if success else self.penalty_incorrect_program_state

                meta = 'expected {}: {}'.format(
                    '{} times'.format(expect_state.times)
                        if expect_state.times >= 0 else 'at least once',
                    expect_state.program_state_text)

                if success:
                    meta = '<g>{}</>'.format(meta)

                maximum_possible_penalty = self.penalty_incorrect_program_state
                maximum_possible_penalty_all += maximum_possible_penalty
                name = expect_state.program_state_text
                penalties[meta] = [PenaltyInstance('{} times'.format(
                    len(expect_state.encounters)), p)]
            self.penalties['expected program states'] = PenaltyCommand(
                penalties, maximum_possible_penalty_all)
        except KeyError:
            pass

        # Get the total number of each step kind.
        step_kind_counts = defaultdict(int)
        for step in getattr(steps, 'steps'):
            step_kind_counts[step.step_kind] += 1

        # Get DexExpectStepKind results.
        penalties = defaultdict(list)
        maximum_possible_penalty_all = 0
        try:
            for command in steps.commands['DexExpectStepKind']:
                command.eval()
                # Cap the penalty at 2 * expected count or else 1
                maximum_possible_penalty = max(command.count * 2, 1)
                p = abs(command.count - step_kind_counts[command.name])
                actual_penalty = min(p, maximum_possible_penalty)
                key = ('{}'.format(command.name)
                       if actual_penalty else '<g>{}</>'.format(command.name))
                penalties[key] = [PenaltyInstance(p, actual_penalty)]
                maximum_possible_penalty_all += maximum_possible_penalty
            self.penalties['step kind differences'] = PenaltyCommand(
                penalties, maximum_possible_penalty_all)
        except KeyError:
            pass

        if 'DexUnreachable' in steps.commands:
            cmds = steps.commands['DexUnreachable']
            unreach_count = 0

            # Find steps with unreachable in them
            ureachs = [
                s for s in steps.steps if 'DexUnreachable' in s.watches.keys()
            ]

            # There's no need to match up cmds with the actual watches
            upen = self.penalty_unreachable

            count = upen * len(ureachs)
            if count != 0:
                d = dict()
                for x in ureachs:
                    msg = 'line {} reached'.format(x.current_location.lineno)
                    d[msg] = [PenaltyInstance(upen, upen)]
            else:
                d = {
                    '<g>No unreachable lines seen</>': [PenaltyInstance(0, 0)]
                }
            total = PenaltyCommand(d, len(cmds) * upen)

            self.penalties['unreachable lines'] = total

        if 'DexExpectStepOrder' in steps.commands:
            cmds = steps.commands['DexExpectStepOrder']

            # Form a list of which line/cmd we _should_ have seen
            cmd_num_lst = [(x, c.lineno) for c in cmds
                                         for x in c.sequence]
            # Order them by the sequence number
            cmd_num_lst.sort(key=lambda t: t[0])
            # Strip out sequence key
            cmd_num_lst = [y for x, y in cmd_num_lst]

            # Now do the same, but for the actually observed lines/cmds
            ss = steps.steps
            deso = [s for s in ss if 'DexExpectStepOrder' in s.watches.keys()]
            deso = [s.watches['DexExpectStepOrder'] for s in deso]
            # We rely on the steps remaining in order here
            order_list = [int(x.expression) for x in deso]

            # First off, check to see whether or not there are missing items
            expected = Counter(cmd_num_lst)
            seen = Counter(order_list)

            unseen_line_dict = dict()
            skipped_line_dict = dict()

            mispen = self.penalty_missing_step
            num_missing = 0
            num_repeats = 0
            for k, v in expected.items():
                if k not in seen:
                    msg = 'Line {} not seen'.format(k)
                    unseen_line_dict[msg] = [PenaltyInstance(mispen, mispen)]
                    num_missing += v
                elif v > seen[k]:
                    msg = 'Line {} skipped at least once'.format(k)
                    skipped_line_dict[msg] = [PenaltyInstance(mispen, mispen)]
                    num_missing += v - seen[k]
                elif v < seen[k]:
                    # Don't penalise unexpected extra sightings of a line
                    # for now
                    num_repeats = seen[k] - v
                    pass

            if len(unseen_line_dict) == 0:
                pi = PenaltyInstance(0, 0)
                unseen_line_dict['<g>All lines were seen</>'] = [pi]

            if len(skipped_line_dict) == 0:
                pi = PenaltyInstance(0, 0)
                skipped_line_dict['<g>No lines were skipped</>'] = [pi]

            total = PenaltyCommand(unseen_line_dict, len(expected) * mispen)
            self.penalties['Unseen lines'] = total
            total = PenaltyCommand(skipped_line_dict, len(expected) * mispen)
            self.penalties['Skipped lines'] = total

            ordpen = self.penalty_misordered_steps
            cmd_num_lst = [str(x) for x in cmd_num_lst]
            order_list = [str(x) for x in order_list]
            lst = list(difflib.Differ().compare(cmd_num_lst, order_list))
            diff_detail = Counter(l[0] for l in lst)

            assert '?' not in diff_detail

            # Diffs are hard to interpret; there are many algorithms for
            # condensing them. Ignore all that, and just print out the changed
            # sequences, it's up to the user to interpret what's going on.

            def filt_lines(s, seg, e, key):
                lst = [s]
                for x in seg:
                    if x[0] == key:
                        lst.append(int(x[2:]))
                lst.append(e)
                return lst

            diff_msgs = dict()

            def reportdiff(start_idx, segment, end_idx):
                msg = 'Order mismatch, expected linenos {}, saw {}'
                expected_linenos = filt_lines(start_idx, segment, end_idx, '-')
                seen_linenos = filt_lines(start_idx, segment, end_idx, '+')
                msg = msg.format(expected_linenos, seen_linenos)
                diff_msgs[msg] = [PenaltyInstance(ordpen, ordpen)]

            # Group by changed segments.
            start_expt_step = 0
            end_expt_step = 0
            to_print_lst = []
            for k, subit in groupby(lst, lambda x: x[0] == ' '):
                if k:  # Whitespace group
                    nochanged = [x for x in subit]
                    end_expt_step = int(nochanged[0][2:])
                    if len(to_print_lst) > 0:
                        reportdiff(start_expt_step, to_print_lst,
                                   end_expt_step)
                    start_expt_step = int(nochanged[-1][2:])
                    to_print_lst = []
                else:  # Diff group, save for printing
                    to_print_lst = [x for x in subit]

            # If there was a dangling different step, print that too.
            if len(to_print_lst) > 0:
                reportdiff(start_expt_step, to_print_lst, '[End]')

            if len(diff_msgs) == 0:
                diff_msgs['<g>No lines misordered</>'] = [
                    PenaltyInstance(0, 0)
                ]
            total = PenaltyCommand(diff_msgs, len(cmd_num_lst) * ordpen)
            self.penalties['Misordered lines'] = total

        return

    def _calculate_expect_watch_penalties(self, c, maximum_possible_penalty):
        penalties = defaultdict(list)

        if c.line_range[0] == c.line_range[-1]:
            line_range = str(c.line_range[0])
        else:
            line_range = '{}-{}'.format(c.line_range[0], c.line_range[-1])

        name = '{}:{} [{}]'.format(
            os.path.basename(c.path), line_range, c.expression)

        num_actual_watches = len(c.expected_watches) + len(
            c.unexpected_watches)

        penalty_available = maximum_possible_penalty

        # Only penalize for missing values if we have actually seen a watch
        # that's returned us an actual value at some point, or if we've not
        # encountered the value at all.
        if num_actual_watches or c.times_encountered == 0:
            for v in c.missing_values:
                current_penalty = min(penalty_available,
                                      self.penalty_missing_values)
                penalty_available -= current_penalty
                penalties['missing values'].append(
                    PenaltyInstance(v, current_penalty))

        for v in c.encountered_values:
            penalties['<g>expected encountered watches</>'].append(
                PenaltyInstance(v, 0))

        penalty_descriptions = [
            (self.penalty_not_evaluatable, c.invalid_watches,
             'could not evaluate'),
            (self.penalty_variable_optimized, c.optimized_out_watches,
             'result optimized away'),
            (self.penalty_misordered_values, c.misordered_watches,
             'misordered result'),
            (self.penalty_irretrievable, c.irretrievable_watches,
             'result could not be retrieved'),
            (self.penalty_incorrect_values, c.unexpected_watches,
             'unexpected result'),
        ]

        for penalty_score, watches, description in penalty_descriptions:
            # We only penalize the encountered issue for each missing value per
            # command but we still want to record each one, so set the penalty
            # to 0 after the threshold is passed.
            times_to_penalize = len(c.missing_values)

            for w in watches:
                times_to_penalize -= 1
                penalty_score = min(penalty_available, penalty_score)
                penalty_available -= penalty_score
                penalties[description].append(
                    PenaltyInstance(w, penalty_score))
                if not times_to_penalize:
                    penalty_score = 0

        return name, penalties

    @property
    def penalty(self):
        result = 0

        maximum_allowed_penalty = 0
        for name, pen_cmd in self.penalties.items():
            maximum_allowed_penalty += pen_cmd.max_penalty
            value = pen_cmd.pen_dict
            for category, inst_list in value.items():
                result += sum(x.the_penalty for x in inst_list)
        return min(result, maximum_allowed_penalty)

    @property
    def max_penalty(self):
        return sum(p_cat.max_penalty for p_cat in self.penalties.values())

    @property
    def score(self):
        try:
            return 1.0 - (self.penalty / float(self.max_penalty))
        except ZeroDivisionError:
            return float('nan')

    @property
    def summary_string(self):
        score = self.score
        isnan = score != score  # pylint: disable=comparison-with-itself
        color = 'g'
        if score < 0.25 or isnan:
            color = 'r'
        elif score < 0.75:
            color = 'y'

        return '<{}>({:.4f})</>'.format(color, score)

    @property
    def verbose_output(self):  # noqa
        string = ''

        # Add address resolutions if present.
        if self.address_resolutions:
            if self.resolved_addresses:
                string += '\nResolved Addresses:\n'
                for addr, res in self.resolved_addresses.items():
                    string += f"  '{addr}': {res}\n"
            if self.unresolved_addresses:
                string += '\n'
                string += f'Unresolved Addresses:\n  {self.unresolved_addresses}\n'

        string += ('\n')
        for command in sorted(self.penalties):
            pen_cmd = self.penalties[command]
            maximum_possible_penalty = pen_cmd.max_penalty
            total_penalty = 0
            lines = []
            for category in sorted(pen_cmd.pen_dict):
                lines.append('    <r>{}</>:\n'.format(category))

                step_value_results = {}
                for result, penalty in pen_cmd.pen_dict[category]:
                    if not isinstance(result, StepValueInfo):
                        continue
                    if result.expected_value not in step_value_results:
                        step_value_results[result.expected_value] = PenaltyLineRanges(result.step_index, penalty)
                    else:
                        step_value_results[result.expected_value].add_step(result.step_index, penalty)

                for value, penalty_line_range in step_value_results.items():
                    text = f'({value}): {penalty_line_range}'
                    total_penalty += penalty_line_range.penalty
                    lines.append('      {}\n'.format(text))

                for result, penalty in pen_cmd.pen_dict[category]:
                    if isinstance(result, StepValueInfo):
                        continue
                    else:
                        text = str(result)
                    if penalty:
                        assert penalty > 0, penalty
                        total_penalty += penalty
                        text += ' <r>[-{}]</>'.format(penalty)
                    lines.append('      {}\n'.format(text))

                lines.append('\n')

            string += ('  <b>{}</> <y>[{}/{}]</>\n'.format(
                command, total_penalty, maximum_possible_penalty))
            for line in lines:
                string += (line)
        string += ('\n')
        return string

    @property
    def resolved_addresses(self):
        return {addr: format_address(res) for addr, res in self.address_resolutions.items() if res is not None}

    @property
    def unresolved_addresses(self):
        return [addr for addr, res in self.address_resolutions.items() if res is None]

    @property
    def penalty_variable_optimized(self):
        return self.context.options.penalty_variable_optimized

    @property
    def penalty_irretrievable(self):
        return self.context.options.penalty_irretrievable

    @property
    def penalty_not_evaluatable(self):
        return self.context.options.penalty_not_evaluatable

    @property
    def penalty_incorrect_values(self):
        return self.context.options.penalty_incorrect_values

    @property
    def penalty_missing_values(self):
        return self.context.options.penalty_missing_values

    @property
    def penalty_misordered_values(self):
        return self.context.options.penalty_misordered_values

    @property
    def penalty_unreachable(self):
        return self.context.options.penalty_unreachable

    @property
    def penalty_missing_step(self):
        return self.context.options.penalty_missing_step

    @property
    def penalty_misordered_steps(self):
        return self.context.options.penalty_misordered_steps

    @property
    def penalty_incorrect_program_state(self):
        return self.context.options.penalty_incorrect_program_state
