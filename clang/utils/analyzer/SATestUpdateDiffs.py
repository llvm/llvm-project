#!/usr/bin/env python

"""
Update reference results for static analyzer.
"""

import SATestBuild

from subprocess import check_call
import os
import sys

Verbose = 1


def runCmd(Command, **kwargs):
    if Verbose:
        print "Executing %s" % Command
    check_call(Command, shell=True, **kwargs)


def updateReferenceResults(ProjName, ProjBuildMode):
    ProjDir = SATestBuild.getProjectDir(ProjName)

    RefResultsPath = os.path.join(
        ProjDir,
        SATestBuild.getSBOutputDirName(IsReferenceBuild=True))
    CreatedResultsPath = os.path.join(
        ProjDir,
        SATestBuild.getSBOutputDirName(IsReferenceBuild=False))

    if not os.path.exists(CreatedResultsPath):
        print >> sys.stderr, "New results not found, was SATestBuild.py "\
                             "previously run?"
        sys.exit(1)

    BuildLogPath = SATestBuild.getBuildLogPath(RefResultsPath)
    Dirname = os.path.dirname(os.path.abspath(BuildLogPath))
    runCmd("mkdir -p '%s'" % Dirname)
    with open(BuildLogPath, "wb+") as PBuildLogFile:
        # Remove reference results: in git, and then again for a good measure
        # with rm, as git might not remove things fully if there are empty
        # directories involved.
        runCmd('git rm -r -q "%s"' % (RefResultsPath,), stdout=PBuildLogFile)
        runCmd('rm -rf "%s"' % (RefResultsPath,), stdout=PBuildLogFile)

        # Replace reference results with a freshly computed once.
        runCmd('cp -r "%s" "%s"' % (CreatedResultsPath, RefResultsPath,),
               stdout=PBuildLogFile)

        # Run cleanup script.
        SATestBuild.runCleanupScript(ProjDir, PBuildLogFile)

        SATestBuild.normalizeReferenceResults(
            ProjDir, RefResultsPath, ProjBuildMode)

        # Clean up the generated difference results.
        SATestBuild.cleanupReferenceResults(RefResultsPath)

        runCmd('git add "%s"' % (RefResultsPath,), stdout=PBuildLogFile)


def main(argv):
    if len(argv) == 2 and argv[1] in ('-h', '--help'):
        print >> sys.stderr, "Update static analyzer reference results based "\
                             "\non the previous run of SATestBuild.py.\n"\
                             "\nN.B.: Assumes that SATestBuild.py was just run"
        sys.exit(1)

    with SATestBuild.projectFileHandler() as f:
        for (ProjName, ProjBuildMode) in SATestBuild.iterateOverProjects(f):
            updateReferenceResults(ProjName, int(ProjBuildMode))


if __name__ == '__main__':
    main(sys.argv)
