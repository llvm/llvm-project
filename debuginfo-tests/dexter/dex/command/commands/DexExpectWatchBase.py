# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""DexExpectWatch base class, holds logic for how to build and process expected
 watch commands.
"""

import abc
import difflib
import os

from dex.command.CommandBase import CommandBase
from dex.command.StepValueInfo import StepValueInfo


class DexExpectWatchBase(CommandBase):
    def __init__(self, *args, **kwargs):
        if len(args) < 2:
            raise TypeError('expected at least two args')

        self.expression = args[0]
        self.values = [str(arg) for arg in args[1:]]
        try:
            on_line = kwargs.pop('on_line')
            self._from_line = on_line
            self._to_line = on_line
        except KeyError:
            self._from_line = kwargs.pop('from_line', 1)
            self._to_line = kwargs.pop('to_line', 999999)
        self._require_in_order = kwargs.pop('require_in_order', True)
        if kwargs:
            raise TypeError('unexpected named args: {}'.format(
                ', '.join(kwargs)))

        # Number of times that this watch has been encountered.
        self.times_encountered = 0

        # We'll pop from this set as we encounter values so anything left at
        # the end can be considered as not having been seen.
        self._missing_values = set(self.values)

        self.misordered_watches = []

        # List of StepValueInfos for any watch that is encountered as invalid.
        self.invalid_watches = []

        # List of StepValueInfo any any watch where we couldn't retrieve its
        # data.
        self.irretrievable_watches = []

        # List of StepValueInfos for any watch that is encountered as having
        # been optimized out.
        self.optimized_out_watches = []

        # List of StepValueInfos for any watch that is encountered that has an
        # expected value.
        self.expected_watches = []

        # List of StepValueInfos for any watch that is encountered that has an
        # unexpected value.
        self.unexpected_watches = []

        super(DexExpectWatchBase, self).__init__()


    def get_watches(self):
        return [self.expression]

    @property
    def line_range(self):
        return list(range(self._from_line, self._to_line + 1))

    @property
    def missing_values(self):
        return sorted(list(self._missing_values))

    @property
    def encountered_values(self):
        return sorted(list(set(self.values) - self._missing_values))


    def resolve_label(self, label_line_pair):
        # from_line and to_line could have the same label.
        label, lineno = label_line_pair
        if self._to_line == label:
            self._to_line = lineno
        if self._from_line == label:
            self._from_line = lineno

    def has_labels(self):
        return len(self.get_label_args()) > 0

    def get_label_args(self):
        return [label for label in (self._from_line, self._to_line)
                      if isinstance(label, str)]

    @abc.abstractmethod
    def _get_expected_field(self, watch):
        """Return a field from watch that this ExpectWatch command is checking.
        """

    def _handle_watch(self, step_info):
        self.times_encountered += 1

        if not step_info.watch_info.could_evaluate:
            self.invalid_watches.append(step_info)
            return

        if step_info.watch_info.is_optimized_away:
            self.optimized_out_watches.append(step_info)
            return

        if step_info.watch_info.is_irretrievable:
            self.irretrievable_watches.append(step_info)
            return

        if step_info.expected_value not in self.values:
            self.unexpected_watches.append(step_info)
            return

        self.expected_watches.append(step_info)
        try:
            self._missing_values.remove(step_info.expected_value)
        except KeyError:
            pass

    def _check_watch_order(self, actual_watches, expected_values):
        """Use difflib to figure out whether the values are in the expected order
        or not.
        """
        differences = []
        actual_values = [w.expected_value for w in actual_watches]
        value_differences = list(difflib.Differ().compare(actual_values,
                                                          expected_values))

        missing_value = False
        index = 0
        for vd in value_differences:
            kind = vd[0]
            if kind == '+':
                # A value that is encountered in the expected list but not in the
                # actual list.  We'll keep a note that something is wrong and flag
                # the next value that matches as misordered.
                missing_value = True
            elif kind == ' ':
                # This value is as expected.  It might still be wrong if we've
                # previously encountered a value that is in the expected list but
                #  not the actual list.
                if missing_value:
                    missing_value = False
                    differences.append(actual_watches[index])
                index += 1
            elif kind == '-':
                # A value that is encountered in the actual list but not the
                #  expected list.
                differences.append(actual_watches[index])
                index += 1
            else:
                assert False, 'unexpected diff:{}'.format(vd)

        return differences

    def eval(self, step_collection):
        for step in step_collection.steps:
            loc = step.current_location

            if (loc.path and os.path.exists(loc.path) and
                os.path.exists(self.path) and
                os.path.samefile(loc.path, self.path) and
                loc.lineno in self.line_range):
                try:
                    watch = step.program_state.frames[0].watches[self.expression]
                except KeyError:
                    pass
                else:
                    expected_field = self._get_expected_field(watch)
                    step_info = StepValueInfo(step.step_index, watch, 
                                              expected_field)
                    self._handle_watch(step_info)

        if self._require_in_order:
            # A list of all watches where the value has changed.
            value_change_watches = []
            prev_value = None
            for watch in self.expected_watches:
                if watch.expected_value != prev_value:
                    value_change_watches.append(watch)
                    prev_value = watch.expected_value

            self.misordered_watches = self._check_watch_order(
                value_change_watches, [
                    v for v in self.values if v in
                    [w.expected_value for w in self.expected_watches]
                ])
