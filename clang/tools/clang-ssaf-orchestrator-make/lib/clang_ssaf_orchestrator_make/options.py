# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import collections

from . import PROGRAM, VERSION

Problem = collections.namedtuple(
    "Problem", "exit_status message details", defaults=((),)
)

USAGE_ERROR = 2
ENVIRONMENT_ERROR = 1


def build_parser():
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        usage="%(prog)s [options] [--] <build command>",
        description="Run an SSAF analysis over a project as it builds. The "
        "build is executed normally; each compilation, archive and link step "
        "is paired with an SSAF step that produces the corresponding summary.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + VERSION)
    parser.add_argument(
        "-o",
        "--output-dir",
        metavar="<dir>",
        default="ssaf-output",
        help="Directory for the SSAF artifacts (default: %(default)s).",
    )
    parser.add_argument(
        "--extract-summaries",
        metavar="<name,...>",
        required=True,
        help="Comma-separated summary extractors to run over every "
        "translation unit. Required.",
    )
    parser.add_argument(
        "--ssaf-clang",
        metavar="<path>",
        default=None,
        help="SSAF-capable clang used for the SSAF compilations "
        "(default: clang on PATH).",
    )
    parser.add_argument(
        "--analysis",
        metavar="<name>",
        action="append",
        default=[],
        dest="analyses",
        help="Whole-program analysis to run on the link unit summary. "
        "Repeatable. Without it the run stops after summary extraction.",
    )
    parser.add_argument(
        "--transformation",
        metavar="<name>",
        default=None,
        help="Source transformation to run in the second pass. Requires "
        "--analysis. Without it the run stops after the analysis.",
    )
    parser.add_argument(
        "--ssaf-linker",
        metavar="<path>",
        default=None,
        help="Path to clang-ssaf-linker (default: found on PATH).",
    )
    parser.add_argument(
        "--ssaf-analyzer",
        metavar="<path>",
        default=None,
        help="Path to clang-ssaf-analyzer (default: found on PATH).",
    )
    parser.add_argument(
        "--ssaf-src-edit-merge",
        metavar="<path>",
        default=None,
        help="Path to clang-ssaf-src-edit-merge (default: found on PATH).",
    )
    parser.add_argument(
        "--ssaf-merge-transformation-report",
        metavar="<path>",
        default=None,
        help="Path to clang-ssaf-merge-transformation-report "
        "(default: found on PATH).",
    )
    parser.add_argument(
        "--ssaf-merge-conflict-report",
        metavar="<path>",
        default=None,
        help="Path to clang-ssaf-merge-conflict-report (default: found on PATH).",
    )
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
        help="Exit successfully even if some SSAF steps failed.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print every command the orchestrator and the shims run.",
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
        output_dir=None,
        extract_summaries=None,
        analyses=(),
        transformation=None,
        ssaf_clang=None,
        ssaf_extra_clang_args=(),
        ssaf_linker=None,
        ssaf_analyzer=None,
        ssaf_src_edit_merge=None,
        ssaf_merge_transformation_report=None,
        ssaf_merge_conflict_report=None,
        keep_going=False,
        verbose=False,
    ):
        self.build_command = list(build_command)
        self.output_dir = output_dir
        self.extract_summaries = extract_summaries
        self.analyses = list(analyses)
        self.transformation = transformation
        self.ssaf_clang = ssaf_clang
        self.ssaf_extra_clang_args = list(ssaf_extra_clang_args)
        self.ssaf_linker = ssaf_linker
        self.ssaf_analyzer = ssaf_analyzer
        self.ssaf_src_edit_merge = ssaf_src_edit_merge
        self.ssaf_merge_transformation_report = ssaf_merge_transformation_report
        self.ssaf_merge_conflict_report = ssaf_merge_conflict_report
        self.keep_going = keep_going
        self.verbose = verbose


def resolve(argv):
    args = build_parser().parse_args(argv)

    build_command = list(args.build)
    if build_command and build_command[0] == "--":
        build_command = build_command[1:]
    if not build_command:
        return None, Problem(USAGE_ERROR, "no build command given")

    if args.transformation and not args.analyses:
        return None, Problem(
            USAGE_ERROR,
            "--transformation needs --analysis: the transformation consumes "
            "the whole-program analysis result",
        )

    return (
        CommandLineOpts(
            build_command=build_command,
            output_dir=args.output_dir,
            extract_summaries=args.extract_summaries,
            analyses=args.analyses,
            transformation=args.transformation,
            ssaf_clang=args.ssaf_clang,
            ssaf_extra_clang_args=args.ssaf_extra_clang_args,
            ssaf_linker=args.ssaf_linker,
            ssaf_analyzer=args.ssaf_analyzer,
            ssaf_src_edit_merge=args.ssaf_src_edit_merge,
            ssaf_merge_transformation_report=args.ssaf_merge_transformation_report,
            ssaf_merge_conflict_report=args.ssaf_merge_conflict_report,
            keep_going=args.keep_going,
            verbose=args.verbose,
        ),
        None,
    )
