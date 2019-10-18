#! /usr/bin/python
#===---------------- Script to generate header files ----------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https:#llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===-----------------------------------------------------------------------===#
#
# This script takes a .h.def file and generates a .h header file.
# See docs/header_generation.md for more information.
#
#===-----------------------------------------------------------------------===#

import argparse
import contextlib
import os
import sys

COMMAND_PREFIX = "%%"
COMMENT_PREFIX = "<!>"

BEGIN_COMMAND = "begin"
COMMENT_COMMAND = "comment"
INCLUDE_FILE_COMMAND = "include_file"


class _Location(object):
    def __init__(self, filename, line_number):
        self.filename = filename
        self.line_number = line_number

    def __str__(self):
        return "%s:%s" % (self.filename, self.line_number)


@contextlib.contextmanager
def output_stream_manager(filename):
    if filename is None:
        try:
            yield sys.stdout
        finally:
            pass
    else:
        output_stream = open(filename, "w")
        try:
            yield output_stream
        finally:
            output_stream.close()


def _parse_command(loc, line):
    open_paren = line.find("(")
    if open_paren < 0 or line[-1] != ")":
        return _fatal_error(loc, "Incorrect header generation command syntax.")
    command_name = line[len(COMMAND_PREFIX):open_paren]
    args = line[open_paren + 1:-1].split(",")
    args = [a.strip() for a in args]
    if len(args) == 1 and not args[0]:
        # There are no args, so we will make the args list an empty list.
        args = []
    return command_name.strip(), args


def _is_named_arg(token):
    if token.startswith("${") and token.endswith("}"):
        return True
    else:
        return False


def _get_arg_name(token):
    return token[2:-1]


def _fatal_error(loc, msg):
    sys.exit("ERROR:%s: %s" % (loc, msg))


def _is_begin_command(line):
    if line.startswith(COMMAND_PREFIX + BEGIN_COMMAND):
        return True


def include_file_command(out_stream, loc, args, values):
    if len(args) != 1:
        _fatal_error(loc, "`%%include_file` command takes exactly one "
                     "argument. %d given." % len(args))
    include_file_path = args[0]
    if _is_named_arg(include_file_path):
        arg_name = _get_arg_name(include_file_path)
        include_file_path = values.get(arg_name)
        if not include_file_path:
            _fatal_error(
                loc,
                "No value specified for argument '%s'." % arg_name)
        if not os.path.exists(include_file_path):
            _fatal_error(
                loc,
                "Include file %s not found." % include_file_path)
    with open(include_file_path, "r") as include_file:
        begin = False
        for line in include_file.readlines():
            line = line.strip()
            if _is_begin_command(line):
                # Parse the command to make sure there are no errors.
                command_name, args = _parse_command(loc, line)
                if args:
                    _fatal_error(loc, "Begin command does not take any args.")
                begin = True
                # Skip the line on which %%begin() is listed.
                continue
            if begin:
                out_stream.write(line + "\n")


def begin_command(out_stream, loc, args, values):
    # "begin" command can only occur in a file included with %%include_file
    # command. It is not a replacement command. Hence, we just fail with
    # a fatal error.
    _fatal_error(loc, "Begin command cannot be listed in an input file.")


# Mapping from a command name to its implementation function.
REPLACEMENT_COMMANDS = {
    INCLUDE_FILE_COMMAND: include_file_command,
    BEGIN_COMMAND: begin_command,
}


def apply_replacement_command(out_stream, loc, line, values):
    if not line.startswith(COMMAND_PREFIX):
        # This line is not a replacement command.
        return line
    command_name, args = _parse_command(loc, line)
    command = REPLACEMENT_COMMANDS.get(command_name.strip())
    if not command:
        _fatal_error(loc, "Unknown replacement command `%`", command_name)
    command(out_stream, loc, args, values)


def parse_options():
    parser = argparse.ArgumentParser(
        description="Script to generate header files from .def files.")
    parser.add_argument("def_file", metavar="DEF_FILE",
                        help="Path to the .def file.")
    parser.add_argument("--args", "-P", nargs= "*", default=[],
                        help="NAME=VALUE pairs for command arguments in the "
                             "input .def file.")
    # The output file argument is optional. If not specified, the generated
    # header file content will be written to stdout.
    parser.add_argument("--out-file", "-o",
                        help="Path to the generated header file. Defaults to "
                             "stdout")
    opts = parser.parse_args()
    if not all(["=" in arg for arg in opts.args]):
        # We want all args to be specified in the form "name=value".
        _fatal_error(
            __file__ + ":" + "[command line]",
            "Command arguments should be listed in the form NAME=VALUE")
    return opts


def main():
    opts = parse_options()
    arg_values = {}
    for name_value_pair in opts.args:
        name, value = name_value_pair.split("=")
        arg_values[name] = value
    with open(opts.def_file, "r") as def_file:
        loc = _Location(opts.def_file, 0)
        with output_stream_manager(opts.out_file) as out_stream:
            for line in def_file:
                loc.line_number += 1
                line = line.strip()
                if line.startswith(COMMAND_PREFIX):
                    replacement_text = apply_replacement_command(
                        out_stream, loc, line, arg_values)
                    out_stream.write("\n")
                elif line.startswith(COMMENT_PREFIX):
                    # Ignore comment line
                    continue
                else:
                    out_stream.write(line + "\n")


if __name__ == "__main__":
    main()
