# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Extended Argument Parser. Extends the argparse module with some extra
functionality, to hopefully aid user-friendliness.
"""

import argparse
import difflib
import unittest

from dex.utils import PrettyOutput
from dex.utils.Exceptions import Error

# re-export all of argparse
for argitem in argparse.__all__:
    vars()[argitem] = getattr(argparse, argitem)


def _did_you_mean(val, possibles):
    close_matches = difflib.get_close_matches(val, possibles)
    did_you_mean = ''
    if close_matches:
        did_you_mean = 'did you mean {}?'.format(' or '.join(
            "<y>'{}'</>".format(c) for c in close_matches[:2]))
    return did_you_mean


def _colorize(message):
    lines = message.splitlines()
    for i, line in enumerate(lines):
        lines[i] = lines[i].replace('usage:', '<g>usage:</>')
        if line.endswith(':'):
            lines[i] = '<g>{}</>'.format(line)
    return '\n'.join(lines)


class ExtArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        """Use the Dexception Error mechanism (including auto-colored output).
        """
        raise Error('{}\n\n{}'.format(message, self.format_usage()))

    # pylint: disable=redefined-builtin
    def _print_message(self, message, file=None):
        if message:
            if file and file.name == '<stdout>':
                file = PrettyOutput.stdout
            else:
                file = PrettyOutput.stderr

            self.context.o.auto(message, file)

    # pylint: enable=redefined-builtin

    def format_usage(self):
        return _colorize(super(ExtArgumentParser, self).format_usage())

    def format_help(self):
        return _colorize(super(ExtArgumentParser, self).format_help() + '\n\n')

    @property
    def _valid_visible_options(self):
        """A list of all non-suppressed command line flags."""
        return [
            item for sublist in vars(self)['_actions']
            for item in sublist.option_strings
            if sublist.help != argparse.SUPPRESS
        ]

    def parse_args(self, args=None, namespace=None):
        """Add 'did you mean' output to errors."""
        args, argv = self.parse_known_args(args, namespace)
        if argv:
            errors = []
            for arg in argv:
                if arg in self._valid_visible_options:
                    error = "unexpected argument: <y>'{}'</>".format(arg)
                else:
                    error = "unrecognized argument: <y>'{}'</>".format(arg)
                    dym = _did_you_mean(arg, self._valid_visible_options)
                    if dym:
                        error += '  ({})'.format(dym)
                errors.append(error)
            self.error('\n       '.join(errors))

        return args

    def add_argument(self, *args, **kwargs):
        """Automatically add the default value to help text."""
        if 'default' in kwargs:
            default = kwargs['default']
            if default is None:
                default = kwargs.pop('display_default', None)

            if (default and isinstance(default, (str, int, float))
                    and default != argparse.SUPPRESS):
                assert (
                    'choices' not in kwargs or default in kwargs['choices']), (
                        "default value '{}' is not one of allowed choices: {}".
                        format(default, kwargs['choices']))
                if 'help' in kwargs and kwargs['help'] != argparse.SUPPRESS:
                    assert isinstance(kwargs['help'], str), type(kwargs['help'])
                    kwargs['help'] = ('{} (default:{})'.format(
                        kwargs['help'], default))

        super(ExtArgumentParser, self).add_argument(*args, **kwargs)

    def __init__(self, context, *args, **kwargs):
        self.context = context
        super(ExtArgumentParser, self).__init__(*args, **kwargs)


class TestExtArgumentParser(unittest.TestCase):
    def test_did_you_mean(self):
        parser = ExtArgumentParser(None)
        parser.add_argument('--foo')
        parser.add_argument('--qoo', help=argparse.SUPPRESS)
        parser.add_argument('jam', nargs='?')

        parser.parse_args(['--foo', '0'])

        expected = (r"^unrecognized argument\: <y>'\-\-doo'</>\s+"
                    r"\(did you mean <y>'\-\-foo'</>\?\)\n"
                    r"\s*<g>usage:</>")
        with self.assertRaisesRegex(Error, expected):
            parser.parse_args(['--doo'])

        parser.add_argument('--noo')

        expected = (r"^unrecognized argument\: <y>'\-\-doo'</>\s+"
                    r"\(did you mean <y>'\-\-noo'</> or <y>'\-\-foo'</>\?\)\n"
                    r"\s*<g>usage:</>")
        with self.assertRaisesRegex(Error, expected):
            parser.parse_args(['--doo'])

        expected = (r"^unrecognized argument\: <y>'\-\-bar'</>\n"
                    r"\s*<g>usage:</>")
        with self.assertRaisesRegex(Error, expected):
            parser.parse_args(['--bar'])

        expected = (r"^unexpected argument\: <y>'\-\-foo'</>\n"
                    r"\s*<g>usage:</>")
        with self.assertRaisesRegex(Error, expected):
            parser.parse_args(['--', 'x', '--foo'])
