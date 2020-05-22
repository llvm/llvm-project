#!/usr/bin/env python

"""
Update reference results for static analyzer.
"""
from __future__ import absolute_import, division, print_function

import SATestBuild

from subprocess import check_call
import os
import sys

Verbose = 0


def runCmd(Command, **kwargs):
    if Verbose:
        print("Executing %s" % Command)
    check_call(Command, shell=True, **kwargs)


def updateReferenceResults(ProjName, ProjBuildMode):
    ProjInfo = SATestBuild.ProjectInfo(ProjName, ProjBuildMode)
    ProjTester = SATestBuild.ProjectTester(ProjInfo)
    ProjDir = ProjTester.get_project_dir()

    ProjTester.is_reference_build = True
    RefResultsPath = os.path.join(ProjDir, ProjTester.get_output_dir())

    ProjTester.is_reference_build = False
    CreatedResultsPath = os.path.join(ProjDir, ProjTester.get_output_dir())

    if not os.path.exists(CreatedResultsPath):
        print("New results not found, was SATestBuild.py "
              "previously run?", file=sys.stderr)
        sys.exit(1)

    BuildLogPath = SATestBuild.get_build_log_path(RefResultsPath)
    Dirname = os.path.dirname(os.path.abspath(BuildLogPath))
    runCmd("mkdir -p '%s'" % Dirname)
    with open(BuildLogPath, "w+") as PBuildLogFile:
        # Remove reference results: in git, and then again for a good measure
        # with rm, as git might not remove things fully if there are empty
        # directories involved.
        runCmd('git rm -r -q "%s"' % (RefResultsPath,), stdout=PBuildLogFile)
        runCmd('rm -rf "%s"' % (RefResultsPath,), stdout=PBuildLogFile)

        # Replace reference results with a freshly computed once.
        runCmd('cp -r "%s" "%s"' % (CreatedResultsPath, RefResultsPath,),
               stdout=PBuildLogFile)

        # Run cleanup script.
        SATestBuild.run_cleanup_script(ProjDir, PBuildLogFile)

        SATestBuild.normalize_reference_results(
            ProjDir, RefResultsPath, ProjBuildMode)

        # Clean up the generated difference results.
        SATestBuild.cleanup_reference_results(RefResultsPath)

        runCmd('git add "%s"' % (RefResultsPath,), stdout=PBuildLogFile)


def main(argv):
    if len(argv) == 2 and argv[1] in ('-h', '--help'):
        print("Update static analyzer reference results based "
              "\non the previous run of SATestBuild.py.\n"
              "\nN.B.: Assumes that SATestBuild.py was just run",
              file=sys.stderr)
        sys.exit(1)

    with open(SATestBuild.get_project_map_path(), "r") as f:
        for ProjName, ProjBuildMode in SATestBuild.get_projects(f):
            updateReferenceResults(ProjName, int(ProjBuildMode))


if __name__ == '__main__':
    main(sys.argv)
