#!/usr/bin/env python

"""
Update reference results for static analyzer.
"""

import SATestBuild

from subprocess import check_call
import os
import sys

Verbose = 1


def runCmd(Command):
    if Verbose:
        print "Executing %s" % Command
    check_call(Command, shell=True)


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

    # Remove reference results: in git, and then again for a good measure
    # with rm, as git might not remove things fully if there are empty
    # directories involved.
    runCmd('git rm -r -q "%s"' % (RefResultsPath,))
    runCmd('rm -rf "%s"' % (RefResultsPath,))

    # Replace reference results with a freshly computed once.
    runCmd('cp -r "%s" "%s"' % (CreatedResultsPath, RefResultsPath,))

    # Run cleanup script.
    BuildLogPath = SATestBuild.getBuildLogPath(RefResultsPath)
    with open(BuildLogPath, "wb+") as PBuildLogFile:
        SATestBuild.runCleanupScript(ProjDir, PBuildLogFile)

    SATestBuild.normalizeReferenceResults(
        ProjDir, RefResultsPath, ProjBuildMode)

    # Clean up the generated difference results.
    SATestBuild.cleanupReferenceResults(RefResultsPath)

    # Remove the created .diffs file before adding.
    removeDiffsSummaryFiles(RefResultsPath)

    runCmd('git add "%s"' % (RefResultsPath,))


def removeDiffsSummaryFiles(RefResultsPath):
    """
    Remove all auto-generated .diffs files in reference data.
    """
    for (Dirpath, Dirnames, Filenames) in os.walk(RefResultsPath):
        if SATestBuild.DiffsSummaryFileName in Filenames:
            runCmd("rm '%s'" % os.path.join(
                Dirpath, SATestBuild.DiffsSummaryFileName))


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
