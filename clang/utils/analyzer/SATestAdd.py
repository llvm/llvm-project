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
                                     run the the download script to download
                                     the project to CachedSource, copy the
                                     CachedSource to another directory (for
                                     example, PatchedSource) and make any
                                     needed modifications to the the copied
                                     source.
                                     Then run:
                                          diff -ur CachedSource PatchedSource \
                                              > changes_for_analyzer.patch
"""
import SATestBuild

import os
import csv
import sys


def isExistingProject(PMapFile, projectID):
    PMapReader = csv.reader(PMapFile)
    for I in PMapReader:
        if projectID == I[0]:
            return True
    return False


def addNewProject(ID, BuildMode):
    """
    Add a new project for testing: build it and add to the Project Map file.
    :param ID: is a short string used to identify a project.
    """

    CurDir = os.path.abspath(os.curdir)
    Dir = SATestBuild.getProjectDir(ID)
    if not os.path.exists(Dir):
        print "Error: Project directory is missing: %s" % Dir
        sys.exit(-1)

    # Build the project.
    SATestBuild.testProject(ID, BuildMode, IsReferenceBuild=True)

    # Add the project ID to the project map.
    ProjectMapPath = os.path.join(CurDir, SATestBuild.ProjectMapFile)

    if os.path.exists(ProjectMapPath):
        FileMode = "r+b"
    else:
        print "Warning: Creating the Project Map file!!"
        FileMode = "w+b"

    with open(ProjectMapPath, FileMode) as PMapFile:
        if (isExistingProject(PMapFile, ID)):
            print >> sys.stdout, 'Warning: Project with ID \'', ID, \
                                 '\' already exists.'
            print >> sys.stdout, "Reference output has been regenerated."
        else:
            PMapWriter = csv.writer(PMapFile)
            PMapWriter.writerow((ID, int(BuildMode)))
            print "The project map is updated: ", ProjectMapPath


# TODO: Add an option not to build.
# TODO: Set the path to the Repository directory.
if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print >> sys.stderr, 'Add a new project for testing to the analyzer'\
                             '\nUsage: ', sys.argv[0],\
                             'project_ID <mode>\n' \
                             'mode: 0 for single file project, ' \
                             '1 for scan_build, ' \
                             '2 for single file c++11 project'
        sys.exit(-1)

    BuildMode = 1
    if (len(sys.argv) >= 3):
        BuildMode = int(sys.argv[2])
    assert((BuildMode == 0) | (BuildMode == 1) | (BuildMode == 2))

    addNewProject(sys.argv[1], BuildMode)
