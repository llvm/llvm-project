# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""This is the main entry point.
It implements some functionality common to all subtools such as command line
parsing and running the unit-testing harnesses, before calling the reequested
subtool.
"""

import imp
import os
import sys

from dex.utils import PrettyOutput, Timer
from dex.utils import ExtArgParse as argparse
from dex.utils import get_root_directory
from dex.utils.Exceptions import Error, ToolArgumentError
from dex.utils.Logging import Logger
from dex.utils.UnitTests import unit_tests_ok
from dex.utils.Version import version
from dex.utils import WorkingDirectory
from dex.utils.ReturnCode import ReturnCode


def _output_bug_report_message(context):
    """In the event of a catastrophic failure, print bug report request to the
    user.
    """
    context.o.red(
        "\n\n"
        "<g>****************************************</>\n"
        "<b>****************************************</>\n"
        "****************************************\n"
        "**                                    **\n"
        "** <y>This is a bug in <a>DExTer</>.</>           **\n"
        "**                                    **\n"
        "**                  <y>Please report it.</> **\n"
        "**                                    **\n"
        "****************************************\n"
        "<b>****************************************</>\n"
        "<g>****************************************</>\n"
        "\n"
        "<b>system:</>\n"
        "<d>{}</>\n\n"
        "<b>version:</>\n"
        "<d>{}</>\n\n"
        "<b>args:</>\n"
        "<d>{}</>\n"
        "\n".format(sys.platform, version("DExTer"), [sys.executable] + sys.argv),
        stream=PrettyOutput.stderr,
    )


def get_tools_directory():
    """Returns directory path where DExTer tool imports can be
    found.
    """
    tools_directory = os.path.join(get_root_directory(), "tools")
    assert os.path.isdir(tools_directory), tools_directory
    return tools_directory


def get_tool_names():
    """Returns a list of expected DExTer Tools"""
    return [
        "help",
        "list-debuggers",
        "no-tool-",
        "run-debugger-internal-",
        "test",
        "view",
    ]


def _set_auto_highlights(context):
    """Flag some strings for auto-highlighting."""
    context.o.auto_reds.extend(
        [
            r"[Ee]rror\:",
            r"[Ee]xception\:",
            r"un(expected|recognized) argument",
        ]
    )
    context.o.auto_yellows.extend(
        [
            r"[Ww]arning\:",
            r"\(did you mean ",
            r"During handling of the above exception, another exception",
        ]
    )


def _get_options_and_args(context):
    """get the options and arguments from the commandline"""
    parser = argparse.ExtArgumentParser(context, add_help=False)
    parser.add_argument("tool", default=None, nargs="?")
    options, args = parser.parse_known_args(sys.argv[1:])

    return options, args


def _get_tool_name(options):
    """get the name of the dexter tool (if passed) specified on the command
    line, otherwise return 'no_tool_'.
    """
    tool_name = options.tool
    if tool_name is None:
        tool_name = "no_tool_"
    else:
        _is_valid_tool_name(tool_name)
    return tool_name


def _is_valid_tool_name(tool_name):
    """check tool name matches a tool directory within the dexter tools
    directory.
    """
    valid_tools = get_tool_names()
    if tool_name not in valid_tools:
        raise Error(
            'invalid tool "{}" (choose from {})'.format(
                tool_name, ", ".join([t for t in valid_tools if not t.endswith("-")])
            )
        )


def _import_tool_module(tool_name):
    """Imports the python module at the tool directory specificed by
    tool_name.
    """
    # format tool argument to reflect tool directory form.
    tool_name = tool_name.replace("-", "_")

    tools_directory = get_tools_directory()
    module_info = imp.find_module(tool_name, [tools_directory])

    return imp.load_module(tool_name, *module_info)


def tool_main(context, tool, args):
    with Timer(tool.name):
        options, defaults = tool.parse_command_line(args)
        Timer.display = options.time_report
        Timer.indent = options.indent_timer_level
        Timer.fn = context.o.blue
        context.options = options
        context.version = version(tool.name)

        if options.version:
            context.o.green("{}\n".format(context.version))
            return ReturnCode.OK

        if options.verbose:
            context.logger.verbosity = 2
        elif options.no_warnings:
            context.logger.verbosity = 0

        if options.unittest != "off" and not unit_tests_ok(context):
            raise Error("<d>unit test failures</>")

        if options.colortest:
            context.o.colortest()
            return ReturnCode.OK

        try:
            tool.handle_base_options(defaults)
        except ToolArgumentError as e:
            raise Error(e)

        dir_ = context.options.working_directory
        with WorkingDirectory(context, dir=dir_) as context.working_directory:
            return_code = tool.go()

        return return_code


class Context(object):
    """Context encapsulates globally useful objects and data; passed to many
    Dexter functions.
    """

    def __init__(self):
        self.o: PrettyOutput = None
        self.logger: Logger = None
        self.working_directory: str = None
        self.options: dict = None
        self.version: str = None
        self.root_directory: str = None


def main() -> ReturnCode:
    context = Context()
    with PrettyOutput() as context.o:
        context.logger = Logger(context.o)
        try:
            context.root_directory = get_root_directory()
            # Flag some strings for auto-highlighting.
            _set_auto_highlights(context)
            options, args = _get_options_and_args(context)
            # raises 'Error' if command line tool is invalid.
            tool_name = _get_tool_name(options)
            module = _import_tool_module(tool_name)
            return tool_main(context, module.Tool(context), args)
        except Error as e:
            context.logger.error(str(e))
            try:
                if context.options.error_debug:
                    raise
            except AttributeError:
                pass
            return ReturnCode._ERROR
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # noqa
            _output_bug_report_message(context)
            raise
