#!/usr/bin/env python

import SATestAdd
import SATestBuild
import SATestUpdateDiffs
import CmpRuns

from ProjectMap import ProjectInfo, ProjectMap

import argparse
import sys
import os

from subprocess import call

SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECTS_DIR = os.path.join(SCRIPTS_DIR, "projects")
DEFAULT_LLVM_DIR = os.path.realpath(os.path.join(SCRIPTS_DIR,
                                                 os.path.pardir,
                                                 os.path.pardir,
                                                 os.path.pardir))


def add(parser, args):
    if args.source == "git" and (args.origin == "" or args.commit == ""):
        parser.error(
            "Please provide both --origin and --commit if source is 'git'")

    if args.source != "git" and (args.origin != "" or args.commit != ""):
        parser.error("Options --origin and --commit don't make sense when "
                     "source is not 'git'")

    project = ProjectInfo(args.name[0], args.mode, args.source, args.origin,
                          args.commit)

    SATestAdd.add_new_project(project)


def build(parser, args):
    SATestBuild.VERBOSE = args.verbose

    project_map = ProjectMap()
    projects = project_map.projects

    if args.projects:
        projects_arg = args.projects.split(",")
        available_projects = [project.name
                              for project in projects]

        # validate that given projects are present in the project map file
        for manual_project in projects_arg:
            if manual_project not in available_projects:
                parser.error("Project '{project}' is not found in "
                             "the project map file. Available projects are "
                             "{all}.".format(project=manual_project,
                                             all=available_projects))

        projects = [project.with_fields(enabled=project.name in projects_arg)
                    for project in projects]

    tester = SATestBuild.RegressionTester(args.jobs,
                                          projects,
                                          args.override_compiler,
                                          args.extra_analyzer_config,
                                          args.regenerate,
                                          args.strictness)
    tests_passed = tester.test_all()

    if not tests_passed:
        sys.stderr.write("ERROR: Tests failed.\n")
        sys.exit(42)


def compare(parser, args):
    dir_old = CmpRuns.ResultsDirectory(args.old[0], args.root_old)
    dir_new = CmpRuns.ResultsDirectory(args.new[0], args.root_new)

    CmpRuns.dump_scan_build_results_diff(dir_old, dir_new,
                                         show_stats=args.show_stats,
                                         stats_only=args.stats_only,
                                         histogram=args.histogram,
                                         verbose_log=args.verbose_log)


def update(parser, args):
    project_map = ProjectMap()
    for project in project_map.projects:
        SATestUpdateDiffs.update_reference_results(project)


def docker(parser, args):
    if len(args.rest) > 0:
        if args.rest[0] != "--":
            parser.error("REST arguments should start with '--'")
        args.rest = args.rest[1:]

    if args.build_image:
        docker_build_image()
    else:
        docker_run(args)


def docker_build_image():
    sys.exit(call("docker build --tag satest-image {}".format(SCRIPTS_DIR),
                  shell=True))


def docker_run(args):
    sys.exit(call("docker run --rm --name satest "
                  "-v {llvm}:/llvm-project "
                  "-v {build}:/build "
                  "-v {clang}:/analyzer "
                  "-v {scripts}:/scripts "
                  "-v {projects}:/projects "
                  "satest-image:latest {args}"
                  .format(llvm=args.llvm_project_dir,
                          build=args.build_dir,
                          clang=args.clang_dir,
                          scripts=SCRIPTS_DIR,
                          projects=PROJECTS_DIR,
                          args=' '.join(args.rest)),
                  shell=True))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # add subcommand
    add_parser = subparsers.add_parser(
        "add",
        help="Add a new project for the analyzer testing.")
    # TODO: Add an option not to build.
    # TODO: Set the path to the Repository directory.
    add_parser.add_argument("name", nargs=1, help="Name of the new project")
    add_parser.add_argument("--mode", action="store", default=1, type=int,
                            choices=[0, 1, 2],
                            help="Build mode: 0 for single file project, "
                            "1 for scan_build, "
                            "2 for single file c++11 project")
    add_parser.add_argument("--source", action="store", default="script",
                            choices=["script", "git", "zip"],
                            help="Source type of the new project: "
                            "'git' for getting from git "
                            "(please provide --origin and --commit), "
                            "'zip' for unpacking source from a zip file, "
                            "'script' for downloading source by running "
                            "a custom script {}"
                            .format(SATestBuild.DOWNLOAD_SCRIPT))
    add_parser.add_argument("--origin", action="store", default="",
                            help="Origin link for a git repository")
    add_parser.add_argument("--commit", action="store", default="",
                            help="Git hash for a commit to checkout")
    add_parser.set_defaults(func=add)

    # build subcommand
    build_parser = subparsers.add_parser(
        "build",
        help="Build projects from the project map and compare results with "
        "the reference.")
    build_parser.add_argument("--strictness", dest="strictness",
                              type=int, default=0,
                              help="0 to fail on runtime errors, 1 to fail "
                              "when the number of found bugs are different "
                              "from the reference, 2 to fail on any "
                              "difference from the reference. Default is 0.")
    build_parser.add_argument("-r", dest="regenerate", action="store_true",
                              default=False,
                              help="Regenerate reference output.")
    build_parser.add_argument("--override-compiler", action="store_true",
                              default=False, help="Call scan-build with "
                              "--override-compiler option.")
    build_parser.add_argument("-j", "--jobs", dest="jobs",
                              type=int, default=0,
                              help="Number of projects to test concurrently")
    build_parser.add_argument("--extra-analyzer-config",
                              dest="extra_analyzer_config", type=str,
                              default="",
                              help="Arguments passed to to -analyzer-config")
    build_parser.add_argument("--projects", action="store", default="",
                              help="Comma-separated list of projects to test")
    build_parser.add_argument("-v", "--verbose", action="count", default=0)
    build_parser.set_defaults(func=build)

    # compare subcommand
    cmp_parser = subparsers.add_parser(
        "compare",
        help="Comparing two static analyzer runs in terms of "
        "reported warnings and execution time statistics.")
    cmp_parser.add_argument("--root-old", dest="root_old",
                            help="Prefix to ignore on source files for "
                            "OLD directory",
                            action="store", type=str, default="")
    cmp_parser.add_argument("--root-new", dest="root_new",
                            help="Prefix to ignore on source files for "
                            "NEW directory",
                            action="store", type=str, default="")
    cmp_parser.add_argument("--verbose-log", dest="verbose_log",
                            help="Write additional information to LOG "
                            "[default=None]",
                            action="store", type=str, default=None,
                            metavar="LOG")
    cmp_parser.add_argument("--stats-only", action="store_true",
                            dest="stats_only", default=False,
                            help="Only show statistics on reports")
    cmp_parser.add_argument("--show-stats", action="store_true",
                            dest="show_stats", default=False,
                            help="Show change in statistics")
    cmp_parser.add_argument("--histogram", action="store", default=None,
                            choices=[CmpRuns.HistogramType.RELATIVE.value,
                                     CmpRuns.HistogramType.LOG_RELATIVE.value,
                                     CmpRuns.HistogramType.ABSOLUTE.value],
                            help="Show histogram of paths differences. "
                            "Requires matplotlib")
    cmp_parser.add_argument("old", nargs=1, help="Directory with old results")
    cmp_parser.add_argument("new", nargs=1, help="Directory with new results")
    cmp_parser.set_defaults(func=compare)

    # update subcommand
    upd_parser = subparsers.add_parser(
        "update",
        help="Update static analyzer reference results based on the previous "
        "run of SATest build. Assumes that SATest build was just run.")
    # TODO: add option to decide whether we should use git
    upd_parser.set_defaults(func=update)

    # docker subcommand
    dock_parser = subparsers.add_parser(
        "docker",
        help="Run regression system in the docker.")

    dock_parser.add_argument("--build-image", action="store_true",
                             help="Build docker image for running tests.")
    dock_parser.add_argument("--llvm-project-dir", action="store",
                             default=DEFAULT_LLVM_DIR,
                             help="Path to LLVM source code. Defaults "
                             "to the repo where this script is located. ")
    dock_parser.add_argument("--build-dir", action="store", default="",
                             help="Path to a directory where docker should "
                             "build LLVM code.")
    dock_parser.add_argument("--clang-dir", action="store", default="",
                             help="Path to find/install LLVM installation.")
    dock_parser.add_argument("rest", nargs=argparse.REMAINDER, default=[],
                             help="Additionall args that will be forwarded "
                             "to the docker's entrypoint.")
    dock_parser.set_defaults(func=docker)

    args = parser.parse_args()
    args.func(parser, args)


if __name__ == "__main__":
    main()
