"""
Test lldb-mi startup options.
"""

from __future__ import print_function

import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiStartupOptionsTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_executable_option_file(self):
        """Test that 'lldb-mi --interpreter %s' loads executable file."""

        self.spawnLldbMi(args="%s" % self.myexe)

        # Test that the executable is loaded when file was specified
        self.expect("-file-exec-and-symbols \"%s\"" % self.myexe)
        self.expect("\^done")

        # Test that lldb-mi is ready when executable was loaded
        self.expect(self.child_prompt, exactly=True)

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Continue
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_executable_option_unknown_file(self):
        """Test that 'lldb-mi --interpreter %s' fails on unknown executable file."""

        # Prepare path to executable
        path = "unknown_file"

        self.spawnLldbMi(args="%s" % path)

        # Test that the executable isn't loaded when unknown file was specified
        self.expect("-file-exec-and-symbols \"%s\"" % path)
        self.expect(
            "\^error,msg=\"Command 'file-exec-and-symbols'. Target binary '%s' is invalid. error: unable to find executable for '%s'\"" %
            (path, path))

        # Test that lldb-mi is ready when executable was loaded
        self.expect(self.child_prompt, exactly=True)

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_executable_option_absolute_path(self):
        """Test that 'lldb-mi --interpreter %s' loads executable which is specified via absolute path."""

        # Prepare path to executable
        import os
        path = os.path.join(os.getcwd(), self.myexe)

        self.spawnLldbMi(args="%s" % path)

        # Test that the executable is loaded when file was specified using
        # absolute path
        self.expect("-file-exec-and-symbols \"%s\"" % path)
        self.expect("\^done")

        # Test that lldb-mi is ready when executable was loaded
        self.expect(self.child_prompt, exactly=True)

        # Run
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_executable_option_relative_path(self):
        """Test that 'lldb-mi --interpreter %s' loads executable which is specified via relative path."""

        # Prepare path to executable
        path = "./%s" % self.myexe

        self.spawnLldbMi(args="%s" % path)

        # Test that the executable is loaded when file was specified using
        # relative path
        self.expect("-file-exec-and-symbols \"%s\"" % path)
        self.expect("\^done")

        # Test that lldb-mi is ready when executable was loaded
        self.expect(self.child_prompt, exactly=True)

        # Run
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_executable_option_unknown_path(self):
        """Test that 'lldb-mi --interpreter %s' fails on executable file which is specified via unknown path."""

        # Prepare path to executable
        path = "unknown_dir/%s" % self.myexe

        self.spawnLldbMi(args="%s" % path)

        # Test that the executable isn't loaded when file was specified using
        # unknown path
        self.expect("-file-exec-and-symbols \"%s\"" % path)
        self.expect(
            "\^error,msg=\"Command 'file-exec-and-symbols'. Target binary '%s' is invalid. error: unable to find executable for '%s'\"" %
            (path, path))

        # Test that lldb-mi is ready when executable was loaded
        self.expect(self.child_prompt, exactly=True)

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux  # llvm.org/pr22841: lldb-mi tests fail on all Linux buildbots
    def test_lldbmi_source_option_start_script(self):
        """Test that 'lldb-mi --interpreter' can execute user's commands after initial commands were executed."""

        # Prepared source file
        sourceFile = "start_script"

        self.spawnLldbMi(args="--source %s" % sourceFile)

        # After '-file-exec-and-symbols a.out'
        self.expect("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # After '-break-insert -f main'
        self.expect("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")

        # After '-exec-run'
        self.expect("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # After '-break-insert main.cpp:BP_return'
        line = line_number('main.cpp', '//BP_return')
        self.expect("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"2\"")

        # After '-exec-continue'
        self.expect("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that lldb-mi is ready after execution of --source start_script
        self.expect(self.child_prompt, exactly=True)

        # Try to evaluate 'a' expression
        self.runCmd("-data-evaluate-expression a")
        self.expect("\^done,value=\"10\"")
        self.expect(self.child_prompt, exactly=True)

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux  # llvm.org/pr22841: lldb-mi tests fail on all Linux buildbots
    def test_lldbmi_source_option_start_script_exit(self):
        """Test that 'lldb-mi --interpreter' can execute a prepared file which passed via --source option."""

        # Prepared source file
        sourceFile = "start_script_exit"

        self.spawnLldbMi(args="--source %s" % sourceFile)

        # After '-file-exec-and-symbols a.out'
        self.expect("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # After '-break-insert -f main'
        self.expect("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")

        # After '-exec-run'
        self.expect("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # After '-break-insert main.cpp:BP_return'
        line = line_number('main.cpp', '//BP_return')
        self.expect("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"2\"")

        # After '-exec-continue'
        self.expect("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # After '-data-evaluate-expression a'
        self.expect("-data-evaluate-expression a")
        self.expect("\^done,value=\"10\"")

        # After '-gdb-exit'
        self.expect("-gdb-exit")
        self.expect("\^exit")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_source_option_start_script_error(self):
        """Test that 'lldb-mi --interpreter' stops execution of initial commands in case of error."""

        # Prepared source file
        sourceFile = "start_script_error"

        self.spawnLldbMi(args="--source %s" % sourceFile)

        # After '-file-exec-and-symbols a.out'
        self.expect("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # After '-break-ins -f main'
        self.expect("-break-ins -f main")
        self.expect("\^error")

        # Test that lldb-mi is ready after execution of --source start_script
        self.expect(self.child_prompt, exactly=True)

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_log_option(self):
        """Test that 'lldb-mi --log' creates a log file in the current directory."""

        logDirectory = "."
        self.spawnLldbMi(args="%s --log" % self.myexe)

        # Test that the executable is loaded when file was specified
        self.expect("-file-exec-and-symbols \"%s\"" % self.myexe)
        self.expect("\^done")

        # Test that lldb-mi is ready when executable was loaded
        self.expect(self.child_prompt, exactly=True)

        # Run
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

        # Check log file is created
        import glob
        import os
        logFile = glob.glob(logDirectory + "/lldb-mi-*.log")

        if not logFile:
            self.fail("log file not found")

        # Delete log
        for f in logFile:
            os.remove(f)

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_log_directory_option(self):
        """Test that 'lldb-mi --log --log-dir' creates a log file in the directory specified by --log-dir."""

        # Create log in temp directory
        import tempfile
        logDirectory = tempfile.gettempdir()

        self.spawnLldbMi(
            args="%s --log --log-dir=%s" %
            (self.myexe, logDirectory))

        # Test that the executable is loaded when file was specified
        self.expect("-file-exec-and-symbols \"%s\"" % self.myexe)
        self.expect("\^done")

        # Test that lldb-mi is ready when executable was loaded
        self.expect(self.child_prompt, exactly=True)

        # Run
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

        # Check log file is created
        import glob
        import os
        logFile = glob.glob(logDirectory + "/lldb-mi-*.log")

        if not logFile:
            self.fail("log file not found")

        # Delete log
        for f in logFile:
            os.remove(f)
