#!/usr/bin/env python

"""
Update reference results for static analyzer.
"""
import SATestBuild

import os
import shutil
import sys

from subprocess import check_call

Verbose = 0


def update_reference_results(project_name: str, build_mode: int):
    project_info = SATestBuild.ProjectInfo(project_name, build_mode)
    tester = SATestBuild.ProjectTester(project_info)
    project_dir = tester.get_project_dir()

    tester.is_reference_build = True
    ref_results_path = os.path.join(project_dir, tester.get_output_dir())

    tester.is_reference_build = False
    created_results_path = os.path.join(project_dir, tester.get_output_dir())

    if not os.path.exists(created_results_path):
        print("New results not found, was SATestBuild.py previously run?",
              file=sys.stderr)
        sys.exit(1)

    build_log_path = SATestBuild.get_build_log_path(ref_results_path)
    build_log_dir = os.path.dirname(os.path.abspath(build_log_path))

    os.makedirs(build_log_dir)

    with open(build_log_path, "w+") as build_log_file:
        def run_cmd(command: str):
            if Verbose:
                print(f"Executing {command}")
            check_call(command, shell=True, stdout=build_log_file)

        # Remove reference results: in git, and then again for a good measure
        # with rm, as git might not remove things fully if there are empty
        # directories involved.
        run_cmd(f"git rm -r -q '{ref_results_path}'")
        shutil.rmtree(ref_results_path)

        # Replace reference results with a freshly computed once.
        shutil.copytree(created_results_path, ref_results_path, symlinks=True)

        # Run cleanup script.
        SATestBuild.run_cleanup_script(project_dir, build_log_file)

        SATestBuild.normalize_reference_results(
            project_dir, ref_results_path, build_mode)

        # Clean up the generated difference results.
        SATestBuild.cleanup_reference_results(ref_results_path)

        run_cmd(f"git add '{ref_results_path}'")


def main(argv):
    if len(argv) == 2 and argv[1] in ("-h", "--help"):
        print("Update static analyzer reference results based "
              "\non the previous run of SATestBuild.py.\n"
              "\nN.B.: Assumes that SATestBuild.py was just run",
              file=sys.stderr)
        sys.exit(1)

    with open(SATestBuild.get_project_map_path(), "r") as f:
        for project_name, build_mode in SATestBuild.get_projects(f):
            update_reference_results(project_name, int(build_mode))


if __name__ == '__main__':
    main(sys.argv)
