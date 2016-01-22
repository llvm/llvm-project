#!/usr/bin/env python

"""
Static Analyzer qualification infrastructure.

The goal is to test the analyzer against different projects, check for failures,
compare results, and measure performance.

Repository Directory will contain sources of the projects as well as the
information on how to build them and the expected output.
Repository Directory structure:
   - ProjectMap file
   - Historical Performance Data
   - Project Dir1
     - ReferenceOutput
   - Project Dir2
     - ReferenceOutput
   ..
Note that the build tree must be inside the project dir.

To test the build of the analyzer one would:
   - Copy over a copy of the Repository Directory. (TODO: Prefer to ensure that
     the build directory does not pollute the repository to min network traffic).
   - Build all projects, until error. Produce logs to report errors.
   - Compare results.

The files which should be kept around for failure investigations:
   RepositoryCopy/Project DirI/ScanBuildResults
   RepositoryCopy/Project DirI/run_static_analyzer.log

Assumptions (TODO: shouldn't need to assume these.):
   The script is being run from the Repository Directory.
   The compiler for scan-build and scan-build are in the PATH.
   export PATH=/Users/zaks/workspace/c2llvm/build/Release+Asserts/bin:$PATH

For more logging, set the  env variables:
   zaks:TI zaks$ export CCC_ANALYZER_LOG=1
   zaks:TI zaks$ export CCC_ANALYZER_VERBOSE=1

The list of checkers tested are hardcoded in the Checkers variable.
For testing additional checkers, use the SA_ADDITIONAL_CHECKERS environment
variable. It should contain a comma separated list.
"""
import CmpRuns

import os
import csv
import sys
import glob
import math
import shutil
import time
import plistlib
import argparse
from subprocess import check_call, check_output, CalledProcessError

#------------------------------------------------------------------------------
# Helper functions.
#------------------------------------------------------------------------------

def detectCPUs():
    """
    Detects the number of CPUs on a system. Cribbed from pp.
    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"):
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else: # OSX:
            return int(capture(['sysctl', '-n', 'hw.ncpu']))
    # Windows:
    if os.environ.has_key("NUMBER_OF_PROCESSORS"):
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1 # Default

def which(command, paths = None):
   """which(command, [paths]) - Look up the given command in the paths string
   (or the PATH environment variable, if unspecified)."""

   if paths is None:
       paths = os.environ.get('PATH','')

   # Check for absolute match first.
   if os.path.exists(command):
       return command

   # Would be nice if Python had a lib function for this.
   if not paths:
       paths = os.defpath

   # Get suffixes to search.
   # On Cygwin, 'PATHEXT' may exist but it should not be used.
   if os.pathsep == ';':
       pathext = os.environ.get('PATHEXT', '').split(';')
   else:
       pathext = ['']

   # Search the paths...
   for path in paths.split(os.pathsep):
       for ext in pathext:
           p = os.path.join(path, command + ext)
           if os.path.exists(p):
               return p

   return None

# Make sure we flush the output after every print statement.
class flushfile(object):
    def __init__(self, f):
        self.f = f
    def write(self, x):
        self.f.write(x)
        self.f.flush()

sys.stdout = flushfile(sys.stdout)

def getProjectMapPath():
    ProjectMapPath = os.path.join(os.path.abspath(os.curdir),
                                  ProjectMapFile)
    if not os.path.exists(ProjectMapPath):
        print "Error: Cannot find the Project Map file " + ProjectMapPath +\
                "\nRunning script for the wrong directory?"
        sys.exit(-1)
    return ProjectMapPath

def getProjectDir(ID):
    return os.path.join(os.path.abspath(os.curdir), ID)

def getSBOutputDirName(IsReferenceBuild) :
    if IsReferenceBuild == True :
        return SBOutputDirReferencePrefix + SBOutputDirName
    else :
        return SBOutputDirName

#------------------------------------------------------------------------------
# Configuration setup.
#------------------------------------------------------------------------------

# Find Clang for static analysis.
Clang = which("clang", os.environ['PATH'])
if not Clang:
    print "Error: cannot find 'clang' in PATH"
    sys.exit(-1)

# Number of jobs.
Jobs = int(math.ceil(detectCPUs() * 0.75))

# Project map stores info about all the "registered" projects.
ProjectMapFile = "projectMap.csv"

# Names of the project specific scripts.
# The script that downloads the project.
DownloadScript = "download_project.sh"
# The script that needs to be executed before the build can start.
CleanupScript = "cleanup_run_static_analyzer.sh"
# This is a file containing commands for scan-build.
BuildScript = "run_static_analyzer.cmd"

# The log file name.
LogFolderName = "Logs"
BuildLogName = "run_static_analyzer.log"
# Summary file - contains the summary of the failures. Ex: This info can be be
# displayed when buildbot detects a build failure.
NumOfFailuresInSummary = 10
FailuresSummaryFileName = "failures.txt"
# Summary of the result diffs.
DiffsSummaryFileName = "diffs.txt"

# The scan-build result directory.
SBOutputDirName = "ScanBuildResults"
SBOutputDirReferencePrefix = "Ref"

# The name of the directory storing the cached project source. If this directory
# does not exist, the download script will be executed. That script should
# create the "CachedSource" directory and download the project source into it.
CachedSourceDirName = "CachedSource"

# The name of the directory containing the source code that will be analyzed.
# Each time a project is analyzed, a fresh copy of its CachedSource directory
# will be copied to the PatchedSource directory and then the local patches
# in PatchfileName will be applied (if PatchfileName exists).
PatchedSourceDirName = "PatchedSource"

# The name of the patchfile specifying any changes that should be applied
# to the CachedSource before analyzing.
PatchfileName = "changes_for_analyzer.patch"

# The list of checkers used during analyzes.
# Currently, consists of all the non-experimental checkers, plus a few alpha
# checkers we don't want to regress on.
Checkers="alpha.unix.SimpleStream,alpha.security.taint,cplusplus.NewDeleteLeaks,core,cplusplus,deadcode,security,unix,osx"

Verbose = 1

#------------------------------------------------------------------------------
# Test harness logic.
#------------------------------------------------------------------------------

# Run pre-processing script if any.
def runCleanupScript(Dir, PBuildLogFile):
    Cwd = os.path.join(Dir, PatchedSourceDirName)
    ScriptPath = os.path.join(Dir, CleanupScript)
    runScript(ScriptPath, PBuildLogFile, Cwd)

# Run the script to download the project, if it exists.
def runDownloadScript(Dir, PBuildLogFile):
    ScriptPath = os.path.join(Dir, DownloadScript)
    runScript(ScriptPath, PBuildLogFile, Dir)

# Run the provided script if it exists.
def runScript(ScriptPath, PBuildLogFile, Cwd):
    if os.path.exists(ScriptPath):
        try:
            if Verbose == 1:
                print "  Executing: %s" % (ScriptPath,)
            check_call("chmod +x '%s'" % ScriptPath, cwd = Cwd,
                                              stderr=PBuildLogFile,
                                              stdout=PBuildLogFile,
                                              shell=True)
            check_call("'%s'" % ScriptPath, cwd = Cwd, stderr=PBuildLogFile,
                                              stdout=PBuildLogFile,
                                              shell=True)
        except:
            print "Error: Running %s failed. See %s for details." % (ScriptPath,
                PBuildLogFile.name)
            sys.exit(-1)

# Download the project and apply the local patchfile if it exists.
def downloadAndPatch(Dir, PBuildLogFile):
    CachedSourceDirPath = os.path.join(Dir, CachedSourceDirName)

    # If the we don't already have the cached source, run the project's
    # download script to download it.
    if not os.path.exists(CachedSourceDirPath):
      runDownloadScript(Dir, PBuildLogFile)
      if not os.path.exists(CachedSourceDirPath):
        print "Error: '%s' not found after download." % (CachedSourceDirPath)
        exit(-1)

    PatchedSourceDirPath = os.path.join(Dir, PatchedSourceDirName)

    # Remove potentially stale patched source.
    if os.path.exists(PatchedSourceDirPath):
        shutil.rmtree(PatchedSourceDirPath)

    # Copy the cached source and apply any patches to the copy.
    shutil.copytree(CachedSourceDirPath, PatchedSourceDirPath, symlinks=True)
    applyPatch(Dir, PBuildLogFile)

def applyPatch(Dir, PBuildLogFile):
    PatchfilePath = os.path.join(Dir, PatchfileName)
    PatchedSourceDirPath = os.path.join(Dir, PatchedSourceDirName)
    if not os.path.exists(PatchfilePath):
        print "  No local patches."
        return

    print "  Applying patch."
    try:
        check_call("patch -p1 < '%s'" % (PatchfilePath),
                    cwd = PatchedSourceDirPath,
                    stderr=PBuildLogFile,
                    stdout=PBuildLogFile,
                    shell=True)
    except:
        print "Error: Patch failed. See %s for details." % (PBuildLogFile.name)
        sys.exit(-1)

# Build the project with scan-build by reading in the commands and
# prefixing them with the scan-build options.
def runScanBuild(Dir, SBOutputDir, PBuildLogFile):
    BuildScriptPath = os.path.join(Dir, BuildScript)
    if not os.path.exists(BuildScriptPath):
        print "Error: build script is not defined: %s" % BuildScriptPath
        sys.exit(-1)

    AllCheckers = Checkers
    if os.environ.has_key('SA_ADDITIONAL_CHECKERS'):
        AllCheckers = AllCheckers + ',' + os.environ['SA_ADDITIONAL_CHECKERS']

    # Run scan-build from within the patched source directory.
    SBCwd = os.path.join(Dir, PatchedSourceDirName)

    SBOptions = "--use-analyzer " + Clang + " "
    SBOptions += "-plist-html -o '" + SBOutputDir + "' "
    SBOptions += "-enable-checker " + AllCheckers + " "
    SBOptions += "--keep-empty "
    # Always use ccc-analyze to ensure that we can locate the failures
    # directory.
    SBOptions += "--override-compiler "
    try:
        SBCommandFile = open(BuildScriptPath, "r")
        SBPrefix = "scan-build " + SBOptions + " "
        for Command in SBCommandFile:
            Command = Command.strip()
            if len(Command) == 0:
                continue;
            # If using 'make', auto imply a -jX argument
            # to speed up analysis.  xcodebuild will
            # automatically use the maximum number of cores.
            if (Command.startswith("make ") or Command == "make") and \
                "-j" not in Command:
                Command += " -j%d" % Jobs
            SBCommand = SBPrefix + Command
            if Verbose == 1:
                print "  Executing: %s" % (SBCommand,)
            check_call(SBCommand, cwd = SBCwd, stderr=PBuildLogFile,
                                               stdout=PBuildLogFile,
                                               shell=True)
    except:
        print "Error: scan-build failed. See ",PBuildLogFile.name,\
              " for details."
        raise

def hasNoExtension(FileName):
    (Root, Ext) = os.path.splitext(FileName)
    if ((Ext == "")) :
        return True
    return False

def isValidSingleInputFile(FileName):
    (Root, Ext) = os.path.splitext(FileName)
    if ((Ext == ".i") | (Ext == ".ii") |
        (Ext == ".c") | (Ext == ".cpp") |
        (Ext == ".m") | (Ext == "")) :
        return True
    return False

# Get the path to the SDK for the given SDK name. Returns None if
# the path cannot be determined.
def getSDKPath(SDKName):
    if which("xcrun") is None:
        return None

    Cmd = "xcrun --sdk " + SDKName + " --show-sdk-path"
    return check_output(Cmd, shell=True).rstrip()

# Run analysis on a set of preprocessed files.
def runAnalyzePreprocessed(Dir, SBOutputDir, Mode):
    if os.path.exists(os.path.join(Dir, BuildScript)):
        print "Error: The preprocessed files project should not contain %s" % \
               BuildScript
        raise Exception()

    CmdPrefix = Clang + " -cc1 "

    # For now, we assume the preprocessed files should be analyzed
    # with the OS X SDK.
    SDKPath = getSDKPath("macosx")
    if SDKPath is not None:
      CmdPrefix += "-isysroot " + SDKPath + " "

    CmdPrefix += "-analyze -analyzer-output=plist -w "
    CmdPrefix += "-analyzer-checker=" + Checkers +" -fcxx-exceptions -fblocks "

    if (Mode == 2) :
        CmdPrefix += "-std=c++11 "

    PlistPath = os.path.join(Dir, SBOutputDir, "date")
    FailPath = os.path.join(PlistPath, "failures");
    os.makedirs(FailPath);

    for FullFileName in glob.glob(Dir + "/*"):
        FileName = os.path.basename(FullFileName)
        Failed = False

        # Only run the analyzes on supported files.
        if (hasNoExtension(FileName)):
            continue
        if (isValidSingleInputFile(FileName) == False):
            print "Error: Invalid single input file %s." % (FullFileName,)
            raise Exception()

        # Build and call the analyzer command.
        OutputOption = "-o '%s.plist' " % os.path.join(PlistPath, FileName)
        Command = CmdPrefix + OutputOption + ("'%s'" % FileName)
        LogFile = open(os.path.join(FailPath, FileName + ".stderr.txt"), "w+b")
        try:
            if Verbose == 1:
                print "  Executing: %s" % (Command,)
            check_call(Command, cwd = Dir, stderr=LogFile,
                                           stdout=LogFile,
                                           shell=True)
        except CalledProcessError, e:
            print "Error: Analyzes of %s failed. See %s for details." \
                  "Error code %d." % \
                   (FullFileName, LogFile.name, e.returncode)
            Failed = True
        finally:
            LogFile.close()

        # If command did not fail, erase the log file.
        if Failed == False:
            os.remove(LogFile.name);

def buildProject(Dir, SBOutputDir, ProjectBuildMode, IsReferenceBuild):
    TBegin = time.time()

    BuildLogPath = os.path.join(SBOutputDir, LogFolderName, BuildLogName)
    print "Log file: %s" % (BuildLogPath,)
    print "Output directory: %s" %(SBOutputDir, )

    # Clean up the log file.
    if (os.path.exists(BuildLogPath)) :
        RmCommand = "rm '%s'" % BuildLogPath
        if Verbose == 1:
            print "  Executing: %s" % (RmCommand,)
        check_call(RmCommand, shell=True)

    # Clean up scan build results.
    if (os.path.exists(SBOutputDir)) :
        RmCommand = "rm -r '%s'" % SBOutputDir
        if Verbose == 1:
            print "  Executing: %s" % (RmCommand,)
            check_call(RmCommand, shell=True)
    assert(not os.path.exists(SBOutputDir))
    os.makedirs(os.path.join(SBOutputDir, LogFolderName))

    # Open the log file.
    PBuildLogFile = open(BuildLogPath, "wb+")

    # Build and analyze the project.
    try:
        if (ProjectBuildMode == 1):
            downloadAndPatch(Dir, PBuildLogFile)
            runCleanupScript(Dir, PBuildLogFile)
            runScanBuild(Dir, SBOutputDir, PBuildLogFile)
        else:
            runAnalyzePreprocessed(Dir, SBOutputDir, ProjectBuildMode)

        if IsReferenceBuild :
            runCleanupScript(Dir, PBuildLogFile)

            # Make the absolute paths relative in the reference results.
            for (DirPath, Dirnames, Filenames) in os.walk(SBOutputDir):
                for F in Filenames:
                    if (not F.endswith('plist')):
                        continue
                    Plist = os.path.join(DirPath, F)
                    Data = plistlib.readPlist(Plist)
                    PathPrefix = Dir
                    if (ProjectBuildMode == 1):
                        PathPrefix = os.path.join(Dir, PatchedSourceDirName)
                    Paths = [SourceFile[len(PathPrefix)+1:]\
                              if SourceFile.startswith(PathPrefix)\
                              else SourceFile for SourceFile in Data['files']]
                    Data['files'] = Paths
                    plistlib.writePlist(Data, Plist)

    finally:
        PBuildLogFile.close()

    print "Build complete (time: %.2f). See the log for more details: %s" % \
           ((time.time()-TBegin), BuildLogPath)

# A plist file is created for each call to the analyzer(each source file).
# We are only interested on the once that have bug reports, so delete the rest.
def CleanUpEmptyPlists(SBOutputDir):
    for F in glob.glob(SBOutputDir + "/*/*.plist"):
        P = os.path.join(SBOutputDir, F)

        Data = plistlib.readPlist(P)
        # Delete empty reports.
        if not Data['files']:
            os.remove(P)
            continue

# Given the scan-build output directory, checks if the build failed
# (by searching for the failures directories). If there are failures, it
# creates a summary file in the output directory.
def checkBuild(SBOutputDir):
    # Check if there are failures.
    Failures = glob.glob(SBOutputDir + "/*/failures/*.stderr.txt")
    TotalFailed = len(Failures);
    if TotalFailed == 0:
        CleanUpEmptyPlists(SBOutputDir)
        Plists = glob.glob(SBOutputDir + "/*/*.plist")
        print "Number of bug reports (non-empty plist files) produced: %d" %\
           len(Plists)
        return;

    # Create summary file to display when the build fails.
    SummaryPath = os.path.join(SBOutputDir, LogFolderName, FailuresSummaryFileName)
    if (Verbose > 0):
        print "  Creating the failures summary file %s" % (SummaryPath,)

    SummaryLog = open(SummaryPath, "w+")
    try:
        SummaryLog.write("Total of %d failures discovered.\n" % (TotalFailed,))
        if TotalFailed > NumOfFailuresInSummary:
            SummaryLog.write("See the first %d below.\n"
                                                   % (NumOfFailuresInSummary,))
        # TODO: Add a line "See the results folder for more."

        FailuresCopied = NumOfFailuresInSummary
        Idx = 0
        for FailLogPathI in Failures:
            if Idx >= NumOfFailuresInSummary:
                break;
            Idx += 1
            SummaryLog.write("\n-- Error #%d -----------\n" % (Idx,));
            FailLogI = open(FailLogPathI, "r");
            try:
                shutil.copyfileobj(FailLogI, SummaryLog);
            finally:
                FailLogI.close()
    finally:
        SummaryLog.close()

    print "Error: analysis failed. See ", SummaryPath
    sys.exit(-1)

# Auxiliary object to discard stdout.
class Discarder(object):
    def write(self, text):
        pass # do nothing

# Compare the warnings produced by scan-build.
# Strictness defines the success criteria for the test:
#   0 - success if there are no crashes or analyzer failure.
#   1 - success if there are no difference in the number of reported bugs.
#   2 - success if all the bug reports are identical.
def runCmpResults(Dir, Strictness = 0):
    TBegin = time.time()

    RefDir = os.path.join(Dir, SBOutputDirReferencePrefix + SBOutputDirName)
    NewDir = os.path.join(Dir, SBOutputDirName)

    # We have to go one level down the directory tree.
    RefList = glob.glob(RefDir + "/*")
    NewList = glob.glob(NewDir + "/*")

    # Log folders are also located in the results dir, so ignore them.
    RefLogDir = os.path.join(RefDir, LogFolderName)
    if RefLogDir in RefList:
        RefList.remove(RefLogDir)
    NewList.remove(os.path.join(NewDir, LogFolderName))

    if len(RefList) == 0 or len(NewList) == 0:
        return False
    assert(len(RefList) == len(NewList))

    # There might be more then one folder underneath - one per each scan-build
    # command (Ex: one for configure and one for make).
    if (len(RefList) > 1):
        # Assume that the corresponding folders have the same names.
        RefList.sort()
        NewList.sort()

    # Iterate and find the differences.
    NumDiffs = 0
    PairList = zip(RefList, NewList)
    for P in PairList:
        RefDir = P[0]
        NewDir = P[1]

        assert(RefDir != NewDir)
        if Verbose == 1:
            print "  Comparing Results: %s %s" % (RefDir, NewDir)

        DiffsPath = os.path.join(NewDir, DiffsSummaryFileName)
        PatchedSourceDirPath = os.path.join(Dir, PatchedSourceDirName)
        Opts = CmpRuns.CmpOptions(DiffsPath, "", PatchedSourceDirPath)
        # Discard everything coming out of stdout (CmpRun produces a lot of them).
        OLD_STDOUT = sys.stdout
        sys.stdout = Discarder()
        # Scan the results, delete empty plist files.
        NumDiffs, ReportsInRef, ReportsInNew = \
            CmpRuns.dumpScanBuildResultsDiff(RefDir, NewDir, Opts, False)
        sys.stdout = OLD_STDOUT
        if (NumDiffs > 0) :
            print "Warning: %r differences in diagnostics. See %s" % \
                  (NumDiffs, DiffsPath,)
        if Strictness >= 2 and NumDiffs > 0:
            print "Error: Diffs found in strict mode (2)."
            sys.exit(-1)
        elif Strictness >= 1 and ReportsInRef != ReportsInNew:
            print "Error: The number of results are different in strict mode (1)."
            sys.exit(-1)

    print "Diagnostic comparison complete (time: %.2f)." % (time.time()-TBegin)
    return (NumDiffs > 0)

def updateSVN(Mode, ProjectsMap):
    try:
        ProjectsMap.seek(0)
        for I in csv.reader(ProjectsMap):
            ProjName = I[0]
            Path = os.path.join(ProjName, getSBOutputDirName(True))

            if Mode == "delete":
                Command = "svn delete '%s'" % (Path,)
            else:
                Command = "svn add '%s'" % (Path,)

            if Verbose == 1:
                print "  Executing: %s" % (Command,)
            check_call(Command, shell=True)

        if Mode == "delete":
            CommitCommand = "svn commit -m \"[analyzer tests] Remove " \
                            "reference results.\""
        else:
            CommitCommand = "svn commit -m \"[analyzer tests] Add new " \
                            "reference results.\""
        if Verbose == 1:
            print "  Executing: %s" % (CommitCommand,)
        check_call(CommitCommand, shell=True)
    except:
        print "Error: SVN update failed."
        sys.exit(-1)

def testProject(ID, ProjectBuildMode, IsReferenceBuild=False, Dir=None, Strictness = 0):
    print " \n\n--- Building project %s" % (ID,)

    TBegin = time.time()

    if Dir is None :
        Dir = getProjectDir(ID)
    if Verbose == 1:
        print "  Build directory: %s." % (Dir,)

    # Set the build results directory.
    RelOutputDir = getSBOutputDirName(IsReferenceBuild)
    SBOutputDir = os.path.join(Dir, RelOutputDir)

    buildProject(Dir, SBOutputDir, ProjectBuildMode, IsReferenceBuild)

    checkBuild(SBOutputDir)

    if IsReferenceBuild == False:
        runCmpResults(Dir, Strictness)

    print "Completed tests for project %s (time: %.2f)." % \
          (ID, (time.time()-TBegin))

def testAll(IsReferenceBuild = False, UpdateSVN = False, Strictness = 0):
    PMapFile = open(getProjectMapPath(), "rb")
    try:
        # Validate the input.
        for I in csv.reader(PMapFile):
            if (len(I) != 2) :
                print "Error: Rows in the ProjectMapFile should have 3 entries."
                raise Exception()
            if (not ((I[1] == "0") | (I[1] == "1") | (I[1] == "2"))):
                print "Error: Second entry in the ProjectMapFile should be 0" \
                      " (single file), 1 (project), or 2(single file c++11)."
                raise Exception()

        # When we are regenerating the reference results, we might need to
        # update svn. Remove reference results from SVN.
        if UpdateSVN == True:
            assert(IsReferenceBuild == True);
            updateSVN("delete",  PMapFile);

        # Test the projects.
        PMapFile.seek(0)
        for I in csv.reader(PMapFile):
            testProject(I[0], int(I[1]), IsReferenceBuild, None, Strictness)

        # Add reference results to SVN.
        if UpdateSVN == True:
            updateSVN("add",  PMapFile);

    except:
        print "Error occurred. Premature termination."
        raise
    finally:
        PMapFile.close()

if __name__ == '__main__':
    # Parse command line arguments.
    Parser = argparse.ArgumentParser(description='Test the Clang Static Analyzer.')
    Parser.add_argument('--strictness', dest='strictness', type=int, default=0,
                       help='0 to fail on runtime errors, 1 to fail when the number\
                             of found bugs are different from the reference, 2 to \
                             fail on any difference from the reference. Default is 0.')
    Parser.add_argument('-r', dest='regenerate', action='store_true', default=False,
                        help='Regenerate reference output.')
    Parser.add_argument('-rs', dest='update_reference', action='store_true',
                        default=False, help='Regenerate reference output and update svn.')
    Args = Parser.parse_args()

    IsReference = False
    UpdateSVN = False
    Strictness = Args.strictness
    if Args.regenerate:
        IsReference = True
    elif Args.update_reference:
        IsReference = True
        UpdateSVN = True

    testAll(IsReference, UpdateSVN, Strictness)
