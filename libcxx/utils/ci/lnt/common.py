# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from typing import Callable, NamedTuple
import argparse
import json


def is_sha(string: str) -> bool:
    return len(string) == 40 and all(c in '0123456789abcdef' for c in string.lower())


def sha(string: str) -> str:
    """
    An argparse type for a full commit SHA, normalized to lowercase.
    """
    if not is_sha(string):
        raise argparse.ArgumentTypeError(f'expected a full 40-character SHA, got {string!r}')
    return string.lower()


def at_least(minimum: int) -> Callable[[str], int]:
    """
    Return an argparse type that accepts integers no smaller than `minimum`.

    A value that is not an integer at all is rejected by raising too, since argparse
    otherwise describes that failure using the name of the callable it was handed,
    which is an implementation detail.
    """
    def parse(string: str) -> int:
        try:
            value = int(string)
        except ValueError:
            raise argparse.ArgumentTypeError(f'expected an integer, got {string!r}')
        if value < minimum:
            raise argparse.ArgumentTypeError(f'expected an integer no smaller than {minimum}, got {string}')
        return value
    return parse


class Target(NamedTuple):
    """
    A unit of work: one commit benchmarked on one machine.
    """
    commit: str
    machine: str


class WorkItem(NamedTuple):
    """
    One line of a plan as produced by `plan-benchmarks`.

    A plan is a sequence of work items written one JSON object per line. It is the
    interchange format between `plan-benchmarks` and `dispatch-benchmarks`.
    """
    commit: str
    machine: str
    samples_have: int
    samples_want: int
    reason: str = ''

    @staticmethod
    def parse(line: str) -> 'WorkItem':
        """
        Read one line of a plan, validating every field.

        A plan is normally produced by `plan-benchmarks`, but it can also be
        hand-written, so we are careful about validating the data here.
        """
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f'expected a JSON object, got {type(item).__name__}')

        def field(name: str, types, required: bool = True, default=None):
            if name not in item:
                if required:
                    raise ValueError(f'missing required field {name}')
                return default
            value = item[name]
            if not isinstance(value, types):
                raise ValueError(f'{name} has the wrong type: {value!r}')
            return value

        commit = field('commit', str)
        if not is_sha(commit):
            raise ValueError(f'commit is not a full 40-character SHA: {commit!r}')
        machine = field('machine', str)
        if not machine:
            raise ValueError('machine is empty')
        samples_have = field('samples_have', int)
        samples_want = field('samples_want', int)
        if samples_have < 0:
            raise ValueError(f'samples_have is negative: {samples_have}')
        if samples_want <= samples_have:
            raise ValueError(f'samples_want {samples_want} is not more than samples_have '
                             f'{samples_have}, so there is nothing to request')
        return WorkItem(commit=commit.lower(),
                        machine=machine,
                        samples_have=samples_have,
                        samples_want=samples_want,
                        reason=field('reason', str, required=False, default=''))

    def serialize(self) -> str:
        """Render the item as the single-line JSON object that a plan is made of."""
        return json.dumps(self._asdict(), sort_keys=True)

    @property
    def target(self) -> Target:
        """The unit of work this item is about."""
        return Target(self.commit, self.machine)

    @property
    def runs_to_request(self) -> int:
        """How many more runs this item is asking for."""
        return self.samples_want - self.samples_have
