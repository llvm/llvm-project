# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Base class for all subtools."""

import abc
import os
import tempfile

from dex import __version__
from dex.utils import ExtArgParse
from dex.utils import PrettyOutput
from dex.utils.ReturnCode import ReturnCode


class ToolBase(object, metaclass=abc.ABCMeta):
    def __init__(self, context):
        self.context = context
        self.parser = None

    @abc.abstractproperty
    def name(self):
        pass

    @abc.abstractmethod
    def add_tool_arguments(self, parser, defaults):
        pass

    def parse_command_line(self, args):
        """Define two parsers: pparser and self.parser.
        pparser deals with args that need to be parsed prior to any of those of
        self.parser.  For example, any args which may affect the state of
        argparse error output.
        """

        class defaults(object):
            pass

        pparser = ExtArgParse.ExtArgumentParser(
            self.context, add_help=False, prog=self.name)

        pparser.add_argument(
            '--no-color-output',
            action='store_true',
            default=False,
            help='do not use colored output on stdout/stderr')
        pparser.add_argument(
            '--time-report',
            action='store_true',
            default=False,
            help='display timing statistics')

        self.parser = ExtArgParse.ExtArgumentParser(
            self.context, parents=[pparser], prog=self.name)
        self.parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            default=False,
            help='enable verbose output')
        self.parser.add_argument(
            '-V',
            '--version',
            action='store_true',
            default=False,
            help='display the DExTer version and exit')
        self.parser.add_argument(
            '-w',
            '--no-warnings',
            action='store_true',
            default=False,
            help='suppress warning output')
        self.parser.add_argument(
            '--unittest',
            type=str,
            choices=['off', 'show-failures', 'show-all'],
            default='off',
            help='run the DExTer codebase unit tests')

        suppress = ExtArgParse.SUPPRESS  # pylint: disable=no-member
        self.parser.add_argument(
            '--colortest', action='store_true', default=False, help=suppress)
        self.parser.add_argument(
            '--error-debug', action='store_true', default=False, help=suppress)
        defaults.working_directory = os.path.join(tempfile.gettempdir(),
                                                  'dexter')
        self.parser.add_argument(
            '--indent-timer-level', type=int, default=1, help=suppress)
        self.parser.add_argument(
            '--working-directory',
            type=str,
            metavar='<file>',
            default=None,
            display_default=defaults.working_directory,
            help='location of working directory')
        self.parser.add_argument(
            '--save-temps',
            action='store_true',
            default=False,
            help='save temporary files')

        self.add_tool_arguments(self.parser, defaults)

        # If an error is encountered during pparser, show the full usage text
        # including self.parser options. Strip the preceding 'usage: ' to avoid
        # having it appear twice.
        pparser.usage = self.parser.format_usage().lstrip('usage: ')

        options, args = pparser.parse_known_args(args)

        if options.no_color_output:
            PrettyOutput.stdout.color_enabled = False
            PrettyOutput.stderr.color_enabled = False

        options = self.parser.parse_args(args, namespace=options)
        return options, defaults

    def handle_base_options(self, defaults):
        self.handle_options(defaults)

        options = self.context.options

        if options.working_directory is None:
            options.working_directory = defaults.working_directory

    @abc.abstractmethod
    def handle_options(self, defaults):
        pass

    @abc.abstractmethod
    def go(self) -> ReturnCode:
        pass
