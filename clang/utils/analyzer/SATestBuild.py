#!/usr/bin/env python

"""
Static Analyzer qualification infrastructure.

The goal is to test the analyzer against different projects,
check for failures, compare results, and measure performance.

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
     the build directory does not pollute the repository to min network
     traffic).
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
import SATestUtils

from subprocess import CalledProcessError, check_call
import argparse
import csv
import glob
import logging
import math
import multiprocessing
import os
import plistlib
import shutil
import sys
import threading
import time
import Queue

#------------------------------------------------------------------------------
# Helper functions.
#------------------------------------------------------------------------------

Local = threading.local()
Local.stdout = sys.stdout
Local.stderr = sys.stderr
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')

class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        # Rstrip in order not to write an extra newline.
        self.logger.log(self.log_level, buf.rstrip())

    def flush(self):
        pass

    def fileno(self):
        return 0


def getProjectMapPath():
    ProjectMapPath = os.path.join(os.path.abspath(os.curdir),
                                  ProjectMapFile)
    if not os.path.exists(ProjectMapPath):
        Local.stdout.write("Error: Cannot find the Project Map file "
                           + ProjectMapPath
                           + "\nRunning script for the wrong directory?\n")
        sys.exit(1)
    return ProjectMapPath


def getProjectDir(ID):
    return os.path.join(os.path.abspath(os.curdir), ID)


def getSBOutputDirName(IsReferenceBuild):
    if IsReferenceBuild:
        return SBOutputDirReferencePrefix + SBOutputDirName
    else:
        return SBOutputDirName

#------------------------------------------------------------------------------
# Configuration setup.
#------------------------------------------------------------------------------


# Find Clang for static analysis.
if 'CC' in os.environ:
    Clang = os.environ['CC']
else:
    Clang = SATestUtils.which("clang", os.environ['PATH'])
if not Clang:
    print "Error: cannot find 'clang' in PATH"
    sys.exit(1)

# Number of jobs.
MaxJobs = int(math.ceil(multiprocessing.cpu_count() * 0.75))

# Project map stores info about all the "registered" projects.
ProjectMapFile = "projectMap.csv"

# Names of the project specific scripts.
# The script that downloads the project.
DownloadScript = "download_project.sh"
# The script that needs to be executed before the build can start.
CleanupScript = "cleanup_run_static_analyzer.sh"
# This is a file containing commands for scan-build.
BuildScript = "run_static_analyzer.cmd"

# A comment in a build script which disables wrapping.
NoPrefixCmd = "#NOPREFIX"

# The log file name.
LogFolderName = "Logs"
BuildLogName = "run_static_analyzer.log"
# Summary file - contains the summary of the failures. Ex: This info can be be
# displayed when buildbot detects a build failure.
NumOfFailuresInSummary = 10
FailuresSummaryFileName = "failures.txt"

# The scan-build result directory.
SBOutputDirName = "ScanBuildResults"
SBOutputDirReferencePrefix = "Ref"

# The name of the directory storing the cached project source. If this
# directory does not exist, the download script will be executed.
# That script should create the "CachedSource" directory and download the
# project source into it.
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
Checkers = ",".join([
    "alpha.unix.SimpleStream",
    "alpha.security.taint",
    "cplusplus.NewDeleteLeaks",
    "core",
    "cplusplus",
    "deadcode",
    "security",
    "unix",
    "osx",
    "nullability"
])

Verbose = 0

#------------------------------------------------------------------------------
# Test harness logic.
#------------------------------------------------------------------------------


def runCleanupScript(Dir, PBuildLogFile):
    """
    Run pre-processing script if any.
    """
    Cwd = os.path.join(Dir, PatchedSourceDirName)
    ScriptPath = os.path.join(Dir, CleanupScript)
    SATestUtils.runScript(ScriptPath, PBuildLogFile, Cwd,
                          Stdout=Local.stdout, Stderr=Local.stderr)


def runDownloadScript(Dir, PBuildLogFile):
    """
    Run the script to download the project, if it exists.
    """
    ScriptPath = os.path.join(Dir, DownloadScript)
    SATestUtils.runScript(ScriptPath, PBuildLogFile, Dir,
                          Stdout=Local.stdout, Stderr=Local.stderr)


def downloadAndPatch(Dir, PBuildLogFile):
    """
    Download the project and apply the local patchfile if it exists.
    """
    CachedSourceDirPath = os.path.join(Dir, CachedSourceDirName)

    # If the we don't already have the cached source, run the project's
    # download script to download it.
    if not os.path.exists(CachedSourceDirPath):
        runDownloadScript(Dir, PBuildLogFile)
        if not os.path.exists(CachedSourceDirPath):
            Local.stderr.write("Error: '%s' not found after download.\n" % (
                               CachedSourceDirPath))
            exit(1)

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
        Local.stdout.write("  No local patches.\n")
        return

    Local.stdout.write("  Applying patch.\n")
    try:
        check_call("patch -p1 < '%s'" % (PatchfilePath),
                   cwd=PatchedSourceDirPath,
                   stderr=PBuildLogFile,
                   stdout=PBuildLogFile,
                   shell=True)
    except:
        Local.stderr.write("Error: Patch failed. See %s for details.\n" % (
            PBuildLogFile.name))
        sys.exit(1)


def runScanBuild(Dir, SBOutputDir, PBuildLogFile):
    """
    Build the project with scan-build by reading in the commands and
    prefixing them with the scan-build options.
    """
    BuildScriptPath = os.path.join(Dir, BuildScript)
    if not os.path.exists(BuildScriptPath):
        Local.stderr.write(
            "Error: build script is not defined: %s\n" % BuildScriptPath)
        sys.exit(1)

    AllCheckers = Checkers
    if 'SA_ADDITIONAL_CHECKERS' in os.environ:
        AllCheckers = AllCheckers + ',' + os.environ['SA_ADDITIONAL_CHECKERS']

    # Run scan-build from within the patched source directory.
    SBCwd = os.path.join(Dir, PatchedSourceDirName)

    SBOptions = "--use-analyzer '%s' " % Clang
    SBOptions += "-plist-html -o '%s' " % SBOutputDir
    SBOptions += "-enable-checker " + AllCheckers + " "
    SBOptions += "--keep-empty "
    AnalyzerConfig = [
        ("stable-report-filename", "true"),
        ("serialize-stats", "true"),
    ]

    SBOptions += "-analyzer-config '%s' " % (
        ",".join("%s=%s" % (key, value) for (key, value) in AnalyzerConfig))

    # Always use ccc-analyze to ensure that we can locate the failures
    # directory.
    SBOptions += "--override-compiler "
    ExtraEnv = {}
    try:
        SBCommandFile = open(BuildScriptPath, "r")
        SBPrefix = "scan-build " + SBOptions + " "
        for Command in SBCommandFile:
            Command = Command.strip()
            if len(Command) == 0:
                continue

            # Custom analyzer invocation specified by project.
            # Communicate required information using environment variables
            # instead.
            if Command == NoPrefixCmd:
                SBPrefix = ""
                ExtraEnv['OUTPUT'] = SBOutputDir
                ExtraEnv['CC'] = Clang
                continue

            # If using 'make', auto imply a -jX argument
            # to speed up analysis.  xcodebuild will
            # automatically use the maximum number of cores.
            if (Command.startswith("make ") or Command == "make") and \
                    "-j" not in Command:
                Command += " -j%d" % MaxJobs
            SBCommand = SBPrefix + Command

            if Verbose == 1:
                Local.stdout.write("  Executing: %s\n" % (SBCommand,))
            check_call(SBCommand, cwd=SBCwd,
                       stderr=PBuildLogFile,
                       stdout=PBuildLogFile,
                       env=dict(os.environ, **ExtraEnv),
                       shell=True)
    except CalledProcessError:
        Local.stderr.write("Error: scan-build failed. Its output was: \n")
        PBuildLogFile.seek(0)
        shutil.copyfileobj(PBuildLogFile, Local.stderr)
        sys.exit(1)


def runAnalyzePreprocessed(Dir, SBOutputDir, Mode):
    """
    Run analysis on a set of preprocessed files.
    """
    if os.path.exists(os.path.join(Dir, BuildScript)):
        Local.stderr.write(
            "Error: The preprocessed files project should not contain %s\n" % (
                BuildScript))
        raise Exception()

    CmdPrefix = Clang + " -cc1 "

    # For now, we assume the preprocessed files should be analyzed
    # with the OS X SDK.
    SDKPath = SATestUtils.getSDKPath("macosx")
    if SDKPath is not None:
        CmdPrefix += "-isysroot " + SDKPath + " "

    CmdPrefix += "-analyze -analyzer-output=plist -w "
    CmdPrefix += "-analyzer-checker=" + Checkers
    CmdPrefix += " -fcxx-exceptions -fblocks "

    if (Mode == 2):
        CmdPrefix += "-std=c++11 "

    PlistPath = os.path.join(Dir, SBOutputDir, "date")
    FailPath = os.path.join(PlistPath, "failures")
    os.makedirs(FailPath)

    for FullFileName in glob.glob(Dir + "/*"):
        FileName = os.path.basename(FullFileName)
        Failed = False

        # Only run the analyzes on supported files.
        if SATestUtils.hasNoExtension(FileName):
            continue
        if not SATestUtils.isValidSingleInputFile(FileName):
            Local.stderr.write(
                "Error: Invalid single input file %s.\n" % (FullFileName,))
            raise Exception()

        # Build and call the analyzer command.
        OutputOption = "-o '%s.plist' " % os.path.join(PlistPath, FileName)
        Command = CmdPrefix + OutputOption + ("'%s'" % FileName)
        LogFile = open(os.path.join(FailPath, FileName + ".stderr.txt"), "w+b")
        try:
            if Verbose == 1:
                Local.stdout.write("  Executing: %s\n" % (Command,))
            check_call(Command, cwd=Dir, stderr=LogFile,
                       stdout=LogFile,
                       shell=True)
        except CalledProcessError, e:
            Local.stderr.write("Error: Analyzes of %s failed. "
                               "See %s for details."
                               "Error code %d.\n" % (
                                   FullFileName, LogFile.name, e.returncode))
            Failed = True
        finally:
            LogFile.close()

        # If command did not fail, erase the log file.
        if not Failed:
            os.remove(LogFile.name)


def getBuildLogPath(SBOutputDir):
    return os.path.join(SBOutputDir, LogFolderName, BuildLogName)


def removeLogFile(SBOutputDir):
    BuildLogPath = getBuildLogPath(SBOutputDir)
    # Clean up the log file.
    if (os.path.exists(BuildLogPath)):
        RmCommand = "rm '%s'" % BuildLogPath
        if Verbose == 1:
            Local.stdout.write("  Executing: %s\n" % (RmCommand,))
        check_call(RmCommand, shell=True)


def buildProject(Dir, SBOutputDir, ProjectBuildMode, IsReferenceBuild):
    TBegin = time.time()

    BuildLogPath = getBuildLogPath(SBOutputDir)
    Local.stdout.write("Log file: %s\n" % (BuildLogPath,))
    Local.stdout.write("Output directory: %s\n" % (SBOutputDir, ))

    removeLogFile(SBOutputDir)

    # Clean up scan build results.
    if (os.path.exists(SBOutputDir)):
        RmCommand = "rm -r '%s'" % SBOutputDir
        if Verbose == 1:
            Local.stdout.write("  Executing: %s\n" % (RmCommand,))
            check_call(RmCommand, shell=True, stdout=Local.stdout,
                       stderr=Local.stderr)
    assert(not os.path.exists(SBOutputDir))
    os.makedirs(os.path.join(SBOutputDir, LogFolderName))

    # Build and analyze the project.
    with open(BuildLogPath, "wb+") as PBuildLogFile:
        if (ProjectBuildMode == 1):
            downloadAndPatch(Dir, PBuildLogFile)
            runCleanupScript(Dir, PBuildLogFile)
            runScanBuild(Dir, SBOutputDir, PBuildLogFile)
        else:
            runAnalyzePreprocessed(Dir, SBOutputDir, ProjectBuildMode)

        if IsReferenceBuild:
            runCleanupScript(Dir, PBuildLogFile)
            normalizeReferenceResults(Dir, SBOutputDir, ProjectBuildMode)

    Local.stdout.write("Build complete (time: %.2f). "
                       "See the log for more details: %s\n" % (
                           (time.time() - TBegin), BuildLogPath))


def normalizeReferenceResults(Dir, SBOutputDir, ProjectBuildMode):
    """
    Make the absolute paths relative in the reference results.
    """
    for (DirPath, Dirnames, Filenames) in os.walk(SBOutputDir):
        for F in Filenames:
            if (not F.endswith('plist')):
                continue
            Plist = os.path.join(DirPath, F)
            Data = plistlib.readPlist(Plist)
            PathPrefix = Dir
            if (ProjectBuildMode == 1):
                PathPrefix = os.path.join(Dir, PatchedSourceDirName)
            Paths = [SourceFile[len(PathPrefix) + 1:]
                     if SourceFile.startswith(PathPrefix)
                     else SourceFile for SourceFile in Data['files']]
            Data['files'] = Paths

            # Remove transient fields which change from run to run.
            for Diag in Data['diagnostics']:
                if 'HTMLDiagnostics_files' in Diag:
                    Diag.pop('HTMLDiagnostics_files')
            if 'clang_version' in Data:
                Data.pop('clang_version')

            plistlib.writePlist(Data, Plist)


def CleanUpEmptyPlists(SBOutputDir):
    """
    A plist file is created for each call to the analyzer(each source file).
    We are only interested on the once that have bug reports,
    so delete the rest.
    """
    for F in glob.glob(SBOutputDir + "/*/*.plist"):
        P = os.path.join(SBOutputDir, F)

        Data = plistlib.readPlist(P)
        # Delete empty reports.
        if not Data['files']:
            os.remove(P)
            continue


def CleanUpEmptyFolders(SBOutputDir):
    """
    Remove empty folders from results, as git would not store them.
    """
    Subfolders = glob.glob(SBOutputDir + "/*")
    for Folder in Subfolders:
        if not os.listdir(Folder):
            os.removedirs(Folder)


def checkBuild(SBOutputDir):
    """
    Given the scan-build output directory, checks if the build failed
    (by searching for the failures directories). If there are failures, it
    creates a summary file in the output directory.

    """
    # Check if there are failures.
    Failures = glob.glob(SBOutputDir + "/*/failures/*.stderr.txt")
    TotalFailed = len(Failures)
    if TotalFailed == 0:
        CleanUpEmptyPlists(SBOutputDir)
        CleanUpEmptyFolders(SBOutputDir)
        Plists = glob.glob(SBOutputDir + "/*/*.plist")
        Local.stdout.write(
            "Number of bug reports (non-empty plist files) produced: %d\n" %
            len(Plists))
        return

    Local.stderr.write("Error: analysis failed.\n")
    Local.stderr.write("Total of %d failures discovered.\n" % TotalFailed)
    if TotalFailed > NumOfFailuresInSummary:
        Local.stderr.write(
            "See the first %d below.\n" % NumOfFailuresInSummary)
        # TODO: Add a line "See the results folder for more."

    Idx = 0
    for FailLogPathI in Failures:
        if Idx >= NumOfFailuresInSummary:
            break
        Idx += 1
        Local.stderr.write("\n-- Error #%d -----------\n" % Idx)
        with open(FailLogPathI, "r") as FailLogI:
            shutil.copyfileobj(FailLogI, Local.stdout)

    sys.exit(1)


def runCmpResults(Dir, Strictness=0):
    """
    Compare the warnings produced by scan-build.
    Strictness defines the success criteria for the test:
      0 - success if there are no crashes or analyzer failure.
      1 - success if there are no difference in the number of reported bugs.
      2 - success if all the bug reports are identical.

    :return success: Whether tests pass according to the Strictness
    criteria.
    """
    TestsPassed = True
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

    if len(RefList) != len(NewList):
        print "Mismatch in number of results folders: %s vs %s" % (
            RefList, NewList)
        sys.exit(1)

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
            Local.stdout.write("  Comparing Results: %s %s\n" % (
                               RefDir, NewDir))

        PatchedSourceDirPath = os.path.join(Dir, PatchedSourceDirName)
        Opts, Args = CmpRuns.generate_option_parser().parse_args(
            ["--rootA", "", "--rootB", PatchedSourceDirPath])
        # Scan the results, delete empty plist files.
        NumDiffs, ReportsInRef, ReportsInNew = \
            CmpRuns.dumpScanBuildResultsDiff(RefDir, NewDir, Opts,
                                             deleteEmpty=False,
                                             Stdout=Local.stdout)
        if (NumDiffs > 0):
            Local.stdout.write("Warning: %s differences in diagnostics.\n"
                               % NumDiffs)
        if Strictness >= 2 and NumDiffs > 0:
            Local.stdout.write("Error: Diffs found in strict mode (2).\n")
            TestsPassed = False
        elif Strictness >= 1 and ReportsInRef != ReportsInNew:
            Local.stdout.write("Error: The number of results are different " +
                               " strict mode (1).\n")
            TestsPassed = False

    Local.stdout.write("Diagnostic comparison complete (time: %.2f).\n" % (
                       time.time() - TBegin))
    return TestsPassed


def cleanupReferenceResults(SBOutputDir):
    """
    Delete html, css, and js files from reference results. These can
    include multiple copies of the benchmark source and so get very large.
    """
    Extensions = ["html", "css", "js"]
    for E in Extensions:
        for F in glob.glob("%s/*/*.%s" % (SBOutputDir, E)):
            P = os.path.join(SBOutputDir, F)
            RmCommand = "rm '%s'" % P
            check_call(RmCommand, shell=True)

    # Remove the log file. It leaks absolute path names.
    removeLogFile(SBOutputDir)


class TestProjectThread(threading.Thread):
    def __init__(self, TasksQueue, ResultsDiffer, FailureFlag):
        """
        :param ResultsDiffer: Used to signify that results differ from
        the canonical ones.
        :param FailureFlag: Used to signify a failure during the run.
        """
        self.TasksQueue = TasksQueue
        self.ResultsDiffer = ResultsDiffer
        self.FailureFlag = FailureFlag
        super(TestProjectThread, self).__init__()

        # Needed to gracefully handle interrupts with Ctrl-C
        self.daemon = True

    def run(self):
        while not self.TasksQueue.empty():
            try:
                ProjArgs = self.TasksQueue.get()
                Logger = logging.getLogger(ProjArgs[0])
                Local.stdout = StreamToLogger(Logger, logging.INFO)
                Local.stderr = StreamToLogger(Logger, logging.ERROR)
                if not testProject(*ProjArgs):
                    self.ResultsDiffer.set()
                self.TasksQueue.task_done()
            except:
                self.FailureFlag.set()
                raise


def testProject(ID, ProjectBuildMode, IsReferenceBuild=False, Strictness=0):
    """
    Test a given project.
    :return TestsPassed: Whether tests have passed according
    to the :param Strictness: criteria.
    """
    Local.stdout.write(" \n\n--- Building project %s\n" % (ID,))

    TBegin = time.time()

    Dir = getProjectDir(ID)
    if Verbose == 1:
        Local.stdout.write("  Build directory: %s.\n" % (Dir,))

    # Set the build results directory.
    RelOutputDir = getSBOutputDirName(IsReferenceBuild)
    SBOutputDir = os.path.join(Dir, RelOutputDir)

    buildProject(Dir, SBOutputDir, ProjectBuildMode, IsReferenceBuild)

    checkBuild(SBOutputDir)

    if IsReferenceBuild:
        cleanupReferenceResults(SBOutputDir)
        TestsPassed = True
    else:
        TestsPassed = runCmpResults(Dir, Strictness)

    Local.stdout.write("Completed tests for project %s (time: %.2f).\n" % (
                       ID, (time.time() - TBegin)))
    return TestsPassed


def projectFileHandler():
    return open(getProjectMapPath(), "rb")


def iterateOverProjects(PMapFile):
    """
    Iterate over all projects defined in the project file handler `PMapFile`
    from the start.
    """
    PMapFile.seek(0)
    for I in csv.reader(PMapFile):
        if (SATestUtils.isCommentCSVLine(I)):
            continue
        yield I


def validateProjectFile(PMapFile):
    """
    Validate project file.
    """
    for I in iterateOverProjects(PMapFile):
        if len(I) != 2:
            print "Error: Rows in the ProjectMapFile should have 2 entries."
            raise Exception()
        if I[1] not in ('0', '1', '2'):
            print "Error: Second entry in the ProjectMapFile should be 0" \
                  " (single file), 1 (project), or 2(single file c++11)."
            raise Exception()

def singleThreadedTestAll(ProjectsToTest):
    """
    Run all projects.
    :return: whether tests have passed.
    """
    Success = True
    for ProjArgs in ProjectsToTest:
        Success &= testProject(*ProjArgs)
    return Success

def multiThreadedTestAll(ProjectsToTest, Jobs):
    """
    Run each project in a separate thread.

    This is OK despite GIL, as testing is blocked
    on launching external processes.

    :return: whether tests have passed.
    """
    TasksQueue = Queue.Queue()

    for ProjArgs in ProjectsToTest:
        TasksQueue.put(ProjArgs)

    ResultsDiffer = threading.Event()
    FailureFlag = threading.Event()

    for i in range(Jobs):
        T = TestProjectThread(TasksQueue, ResultsDiffer, FailureFlag)
        T.start()

    # Required to handle Ctrl-C gracefully.
    while TasksQueue.unfinished_tasks:
        time.sleep(0.1)  # Seconds.
        if FailureFlag.is_set():
            Local.stderr.write("Test runner crashed\n")
            sys.exit(1)
    return not ResultsDiffer.is_set()


def testAll(Args):
    ProjectsToTest = []

    with projectFileHandler() as PMapFile:
        validateProjectFile(PMapFile)

        # Test the projects.
        for (ProjName, ProjBuildMode) in iterateOverProjects(PMapFile):
            ProjectsToTest.append((ProjName,
                                  int(ProjBuildMode),
                                  Args.regenerate,
                                  Args.strictness))
    if Args.jobs <= 1:
        return singleThreadedTestAll(ProjectsToTest)
    else:
        return multiThreadedTestAll(ProjectsToTest, Args.jobs)


if __name__ == '__main__':
    # Parse command line arguments.
    Parser = argparse.ArgumentParser(
        description='Test the Clang Static Analyzer.')
    Parser.add_argument('--strictness', dest='strictness', type=int, default=0,
                        help='0 to fail on runtime errors, 1 to fail when the \
                             number of found bugs are different from the \
                             reference, 2 to fail on any difference from the \
                             reference. Default is 0.')
    Parser.add_argument('-r', dest='regenerate', action='store_true',
                        default=False, help='Regenerate reference output.')
    Parser.add_argument('-j', '--jobs', dest='jobs', type=int,
                        default=0,
                        help='Number of projects to test concurrently')
    Args = Parser.parse_args()

    TestsPassed = testAll(Args)
    if not TestsPassed:
        print "ERROR: Tests failed."
        sys.exit(42)
