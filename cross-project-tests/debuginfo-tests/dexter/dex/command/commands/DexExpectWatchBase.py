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
import math
from collections import namedtuple
from pathlib import PurePath

from dex.command.CommandBase import CommandBase, StepExpectInfo
from dex.command.StepValueInfo import StepValueInfo
from dex.utils.Exceptions import NonFloatValueInCommand

class AddressExpression(object):
    def __init__(self, name, offset=0):
        self.name = name
        self.offset = offset

    def is_resolved(self, resolutions):
        return self.name in resolutions

    # Given the resolved value of the address, resolve the final value of
    # this expression.
    def resolved_value(self, resolutions):
        if not self.name in resolutions or resolutions[self.name] is None:
            return None
        # Technically we should fill(8) if we're debugging on a 32bit architecture?
        return format_address(resolutions[self.name] + self.offset)

def format_address(value, address_width=64):
    return "0x" + hex(value)[2:].zfill(math.ceil(address_width/4))

def resolved_value(value, resolutions):
    return value.resolved_value(resolutions) if isinstance(value, AddressExpression) else value

class DexExpectWatchBase(CommandBase):
    def __init__(self, *args, **kwargs):
        if len(args) < 2:
            raise TypeError('expected at least two args')

        self.expression = args[0]
        self.values = [arg if isinstance(arg, AddressExpression) else str(arg) for arg in args[1:]]
        try:
            on_line = kwargs.pop('on_line')
            self._from_line = on_line
            self._to_line = on_line
        except KeyError:
            self._from_line = kwargs.pop('from_line', 1)
            self._to_line = kwargs.pop('to_line', 999999)
        self._require_in_order = kwargs.pop('require_in_order', True)
        self.float_range = kwargs.pop('float_range', None)
        if self.float_range is not None:
            for value in self.values:
                try:
                    float(value)
                except ValueError:
                    raise NonFloatValueInCommand(f'Non-float value \'{value}\' when float_range arg provided')
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

        # List of StepValueInfos for all observed watches that were not
        # invalid, irretrievable, or optimized out (combines expected and
        # unexpected).
        self.observed_watches = []

        # dict of address names to their final resolved values, None until it
        # gets assigned externally.
        self.address_resolutions = None

        super(DexExpectWatchBase, self).__init__()

    def resolve_value(self, value):
        return value.resolved_value(self.address_resolutions) if isinstance(value, AddressExpression) else value

    def describe_value(self, value):
        if isinstance(value, AddressExpression):
            offset = ""
            if value.offset > 0:
                offset = f"+{value.offset}"
            elif value.offset < 0:
                offset = str(value.offset)
            desc =  f"address '{value.name}'{offset}"
            if self.resolve_value(value) is not None:
                desc += f" ({self.resolve_value(value)})"
            return desc
        return value

    def get_watches(self):
        return [StepExpectInfo(self.expression, self.path, 0, range(self._from_line, self._to_line + 1))]

    @property
    def line_range(self):
        return list(range(self._from_line, self._to_line + 1))

    @property
    def missing_values(self):
        return sorted(list(self.describe_value(v) for v in self._missing_values))

    @property
    def encountered_values(self):
        return sorted(list(set(self.describe_value(v) for v in set(self.values) - self._missing_values)))

    @abc.abstractmethod
    def _get_expected_field(self, watch):
        """Return a field from watch that this ExpectWatch command is checking.
        """

    def _match_expected_floating_point(self, value):
        """Checks to see whether value is a float that falls within the
        acceptance range of one of this command's expected float values, and
        returns the expected value if so; otherwise returns the original
        value."""
        try:
            value_as_float = float(value)
        except ValueError:
            return value

        possible_values = self.values
        for expected in possible_values:
          try:
              expected_as_float = float(expected)
              difference = abs(value_as_float - expected_as_float)
              if difference <= self.float_range:
                  return expected
          except ValueError:
              pass
        return value

    def _maybe_fix_float(self, value):
        if self.float_range is not None:
            return self._match_expected_floating_point(value)
        else:
            return value

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

        expected_value = self._maybe_fix_float(step_info.expected_value)

        # Check to see if this value matches with a resolved address.
        matching_address = None
        for v in self.values:
            if (isinstance(v, AddressExpression) and
                    v.name in self.address_resolutions and
                    self.resolve_value(v) == expected_value):
                matching_address = v
                break

        # If this is not an expected value, either a direct value or an address,
        # then this is an unexpected watch.
        if expected_value not in self.values and matching_address is None:
            self.unexpected_watches.append(step_info)
            return

        self.expected_watches.append(step_info)
        value_to_remove = matching_address if matching_address is not None else expected_value
        try:
            self._missing_values.remove(value_to_remove)
        except KeyError:
            pass

    def _check_watch_order(self, actual_watches, expected_values):
        """Use difflib to figure out whether the values are in the expected order
        or not.
        """
        differences = []
        actual_values = [self._maybe_fix_float(w.expected_value) for w in actual_watches]
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

            if (loc.path and self.path and
                PurePath(loc.path) == PurePath(self.path) and
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
            all_expected_values = []
            for watch in self.expected_watches:
                expected_value = self._maybe_fix_float(watch.expected_value)
                all_expected_values.append(expected_value)
                if expected_value != prev_value:
                    value_change_watches.append(watch)
                    prev_value = expected_value

            resolved_values = [self.resolve_value(v) for v in self.values]
            self.misordered_watches = self._check_watch_order(
                value_change_watches, [
                    v for v in resolved_values if v in all_expected_values
                ])
