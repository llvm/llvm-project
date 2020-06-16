#!/usr/bin/env python

"""
Static Analyzer qualification infrastructure: adding a new project to
the Repository Directory.

 Add a new project for testing: build it and add to the Project Map file.
   Assumes it's being run from the Repository Directory.
   The project directory should be added inside the Repository Directory and
   have the same name as the project ID

 The project should use the following files for set up:
      - cleanup_run_static_analyzer.sh - prepare the build environment.
                                     Ex: make clean can be a part of it.
      - run_static_analyzer.cmd - a list of commands to run through scan-build.
                                     Each command should be on a separate line.
                                     Choose from: configure, make, xcodebuild
      - download_project.sh - download the project into the CachedSource/
                                     directory. For example, download a zip of
                                     the project source from GitHub, unzip it,
                                     and rename the unzipped directory to
                                     'CachedSource'. This script is not called
                                     when 'CachedSource' is already present,
                                     so an alternative is to check the
                                     'CachedSource' directory into the
                                     repository directly.
      - CachedSource/ - An optional directory containing the source of the
                                     project being analyzed. If present,
                                     download_project.sh will not be called.
      - changes_for_analyzer.patch - An optional patch file for any local
                                     changes
                                     (e.g., to adapt to newer version of clang)
                                     that should be applied to CachedSource
                                     before analysis. To construct this patch,
                                     run the download script to download
                                     the project to CachedSource, copy the
                                     CachedSource to another directory (for
                                     example, PatchedSource) and make any
                                     needed modifications to the copied
                                     source.
                                     Then run:
                                          diff -ur CachedSource PatchedSource \
                                              > changes_for_analyzer.patch
"""
import SATestBuild
from ProjectMap import ProjectMap, ProjectInfo

import argparse
import os
import sys


def add_new_project(project: ProjectInfo):
    """
    Add a new project for testing: build it and add to the Project Map file.
    :param name: is a short string used to identify a project.
    """

    test_info = SATestBuild.TestInfo(project,
                                     is_reference_build=True)
    tester = SATestBuild.ProjectTester(test_info)

    project_dir = tester.get_project_dir()
    if not os.path.exists(project_dir):
        print(f"Error: Project directory is missing: {project_dir}")
        sys.exit(-1)

    # Build the project.
    tester.test()

    # Add the project name to the project map.
    project_map = ProjectMap(should_exist=False)

    if is_existing_project(project_map, project):
        print(f"Warning: Project with name '{project.name}' already exists.",
              file=sys.stdout)
        print("Reference output has been regenerated.", file=sys.stdout)
    else:
        project_map.projects.append(project)
        project_map.save()


def is_existing_project(project_map: ProjectMap, project: ProjectInfo) -> bool:
    return any(existing_project.name == project.name
               for existing_project in project_map.projects)


# TODO: Add an option not to build.
# TODO: Set the path to the Repository directory.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("name", nargs=1, help="Name of the new project")
    parser.add_argument("--mode", action="store", default=1, type=int,
                        choices=[0, 1, 2],
                        help="Build mode: 0 for single file project, "
                        "1 for scan_build, "
                        "2 for single file c++11 project")
    parser.add_argument("--source", action="store", default="script",
                        choices=["script", "git", "zip"],
                        help=f"Source type of the new project: "
                        f"'git' for getting from git "
                        f"(please provide --origin and --commit), "
                        f"'zip' for unpacking source from a zip file, "
                        f"'script' for downloading source by running "
                        f"a custom script {SATestBuild.DOWNLOAD_SCRIPT}")
    parser.add_argument("--origin", action="store", default="",
                        help="Origin link for a git repository")
    parser.add_argument("--commit", action="store", default="",
                        help="Git hash for a commit to checkout")

    args = parser.parse_args()

    if args.source == "git" and (args.origin == "" or args.commit == ""):
        parser.error(
            "Please provide both --origin and --commit if source is 'git'")

    if args.source != "git" and (args.origin != "" or args.commit != ""):
        parser.error("Options --origin and --commit don't make sense when "
                     "source is not 'git'")

    project = ProjectInfo(args.name[0], args.mode, args.source, args.origin,
                          args.commit)

    add_new_project(project)
