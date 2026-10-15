# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import collections

from . import PROGRAM, VERSION
from .transformations import TRANSFORMATIONS

Problem = collections.namedtuple(
    "Problem", "exit_status message details", defaults=((),)
)

USAGE_ERROR = 2
ENVIRONMENT_ERROR = 1

FORWARDED_TOOLS = (
    (
        "--ssaf-clang",
        "SSAF-capable clang used for the SSAF compilations (default: clang on PATH).",
    ),
    ("--ssaf-linker", "Path to clang-ssaf-linker (default: found on PATH)."),
    ("--ssaf-analyzer", "Path to clang-ssaf-analyzer (default: found on PATH)."),
    (
        "--ssaf-src-edit-merge",
        "Path to clang-ssaf-src-edit-merge (default: found on PATH).",
    ),
    (
        "--ssaf-report-merge",
        "Path to clang-ssaf-report-merge (default: found on PATH).",
    ),
    (
        "--ssaf-conflict-merge",
        "Path to clang-ssaf-conflict-merge (default: found on PATH).",
    ),
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        usage="%(prog)s [options] --transformation=<name> [--] <build command>",
        description="Run an SSAF transformation over a project as it builds. "
        "The source edits are written as a patch, and the transformation report "
        "is printed and written as SARIF.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + VERSION)
    parser.add_argument(
        "--transformation",
        metavar="<name>",
        required=True,
        choices=sorted(TRANSFORMATIONS),
        help="The transformation to run, one of: %s. Required."
        % ", ".join(sorted(TRANSFORMATIONS)),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        metavar="<dir>",
        default="ssaf-output",
        help="Directory for the results and the SSAF artifacts "
        "(default: %(default)s).",
    )
    for option, purpose in FORWARDED_TOOLS:
        parser.add_argument(option, metavar="<path>", default=None, help=purpose)
    parser.add_argument(
        "--ssaf-extra-clang-arg",
        metavar="<arg>",
        action="append",
        default=[],
        dest="ssaf_extra_clang_args",
        help="Additional argument for every SSAF compilation. Repeatable.",
    )
    parser.add_argument(
        "-k",
        "--keep-going",
        action="store_true",
        help="Carry on even if some SSAF steps failed.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print every command that runs.",
    )
    parser.add_argument(
        "build",
        nargs=argparse.REMAINDER,
        metavar="<build command>",
        help="The build command. A '--' separator before it is optional.",
    )
    return parser


class CommandLineOpts:
    def __init__(
        self,
        build_command=(),
        transformation=None,
        output_dir=None,
        orchestrator_args=(),
        verbose=False,
    ):
        self.build_command = list(build_command)
        self.transformation = transformation
        self.output_dir = output_dir
        self.orchestrator_args = list(orchestrator_args)
        self.verbose = verbose


def resolve(argv):
    args = build_parser().parse_args(argv)

    build_command = list(args.build)
    if build_command and build_command[0] == "--":
        build_command = build_command[1:]
    if not build_command:
        return None, Problem(USAGE_ERROR, "no build command given")

    orchestrator_args = []
    for option, _ in FORWARDED_TOOLS:
        value = getattr(args, option[2:].replace("-", "_"))
        if value is not None:
            orchestrator_args.append("%s=%s" % (option, value))
    for arg in args.ssaf_extra_clang_args:
        orchestrator_args.append("--ssaf-extra-clang-arg=" + arg)
    if args.keep_going:
        orchestrator_args.append("--keep-going")
    if args.verbose:
        orchestrator_args.append("--verbose")

    return (
        CommandLineOpts(
            build_command=build_command,
            transformation=args.transformation,
            output_dir=args.output_dir,
            orchestrator_args=orchestrator_args,
            verbose=args.verbose,
        ),
        None,
    )
