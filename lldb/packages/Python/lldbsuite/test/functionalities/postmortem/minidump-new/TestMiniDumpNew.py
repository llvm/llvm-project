"""
Test basics of Minidump debugging.
"""

from __future__ import print_function
from six import iteritems

import shutil

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiniDumpNewTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    _linux_x86_64_pid = 29917
    _linux_x86_64_not_crashed_pid = 29939
    _linux_x86_64_not_crashed_pid_offset = 0xD967

    def setUp(self):
        super(MiniDumpNewTestCase, self).setUp()
        self._initial_platform = lldb.DBG.GetSelectedPlatform()

    def tearDown(self):
        lldb.DBG.SetSelectedPlatform(self._initial_platform)
        super(MiniDumpNewTestCase, self).tearDown()

    def check_state(self):
        with open(os.devnull) as devnul:
            # sanitize test output
            self.dbg.SetOutputFileHandle(devnul, False)
            self.dbg.SetErrorFileHandle(devnul, False)

            self.assertTrue(self.process.is_stopped)

            # Process.Continue
            error = self.process.Continue()
            self.assertFalse(error.Success())
            self.assertTrue(self.process.is_stopped)

            # Thread.StepOut
            thread = self.process.GetSelectedThread()
            thread.StepOut()
            self.assertTrue(self.process.is_stopped)

            # command line
            self.dbg.HandleCommand('s')
            self.assertTrue(self.process.is_stopped)
            self.dbg.HandleCommand('c')
            self.assertTrue(self.process.is_stopped)

            # restore file handles
            self.dbg.SetOutputFileHandle(None, False)
            self.dbg.SetErrorFileHandle(None, False)

    def test_process_info_in_minidump(self):
        """Test that lldb can read the process information from the Minidump."""
        # target create -c linux-x86_64.dmp
        self.dbg.CreateTarget(None)
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("linux-x86_64.dmp")
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertEqual(self.process.GetNumThreads(), 1)
        self.assertEqual(self.process.GetProcessID(), self._linux_x86_64_pid)
        self.check_state()

    def test_thread_info_in_minidump(self):
        """Test that lldb can read the thread information from the Minidump."""
        # target create -c linux-x86_64.dmp
        self.dbg.CreateTarget(None)
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("linux-x86_64.dmp")
        self.check_state()
        # This process crashed due to a segmentation fault in its
        # one and only thread.
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonSignal)
        stop_description = thread.GetStopDescription(256)
        self.assertTrue("SIGSEGV" in stop_description)

    def test_stack_info_in_minidump(self):
        """Test that we can see a trivial stack in a breakpad-generated Minidump."""
        # target create linux-x86_64 -c linux-x86_64.dmp
        self.dbg.CreateTarget("linux-x86_64")
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("linux-x86_64.dmp")
        self.check_state()
        self.assertEqual(self.process.GetNumThreads(), 1)
        self.assertEqual(self.process.GetProcessID(), self._linux_x86_64_pid)
        thread = self.process.GetThreadAtIndex(0)
        # frame #0: linux-x86_64`crash()
        # frame #1: linux-x86_64`_start
        self.assertEqual(thread.GetNumFrames(), 2)
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid())
        pc = frame.GetPC()
        eip = frame.FindRegister("pc")
        self.assertTrue(eip.IsValid())
        self.assertEqual(pc, eip.GetValueAsUnsigned())

    def test_snapshot_minidump(self):
        """Test that if we load a snapshot minidump file (meaning the process
        did not crash) there is no stop reason."""
        # target create -c linux-x86_64_not_crashed.dmp
        self.dbg.CreateTarget(None)
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("linux-x86_64_not_crashed.dmp")
        self.check_state()
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonNone)
        stop_description = thread.GetStopDescription(256)
        self.assertEqual(stop_description, "")

    def do_test_deeper_stack(self, binary, core, pid):
        target = self.dbg.CreateTarget(binary)
        process = target.LoadCore(core)
        thread = process.GetThreadAtIndex(0)

        self.assertEqual(process.GetProcessID(), pid)

        expected_stack = {1: 'bar', 2: 'foo', 3: '_start'}
        self.assertGreaterEqual(thread.GetNumFrames(), len(expected_stack))
        for index, name in iteritems(expected_stack):
            frame = thread.GetFrameAtIndex(index)
            self.assertTrue(frame.IsValid())
            function_name = frame.GetFunctionName()
            self.assertTrue(name in function_name)

    def test_deeper_stack_in_minidump(self):
        """Test that we can examine a more interesting stack in a Minidump."""
        # Launch with the Minidump, and inspect the stack.
        # target create linux-x86_64_not_crashed -c linux-x86_64_not_crashed.dmp
        self.do_test_deeper_stack("linux-x86_64_not_crashed",
                                  "linux-x86_64_not_crashed.dmp",
                                  self._linux_x86_64_not_crashed_pid)

    def do_change_pid_in_minidump(self, core, newcore, offset, oldpid, newpid):
        """ This assumes that the minidump is breakpad generated on Linux -
        meaning that the PID in the file will be an ascii string part of
        /proc/PID/status which is written in the file
        """
        shutil.copyfile(core, newcore)
        with open(newcore, "rb+") as f:
            f.seek(offset)
            currentpid = f.read(5).decode('utf-8')
            self.assertEqual(currentpid, oldpid)

            f.seek(offset)
            if len(newpid) < len(oldpid):
                newpid += " " * (len(oldpid) - len(newpid))
            newpid += "\n"
            f.write(newpid.encode('utf-8'))

    def test_deeper_stack_in_minidump_with_same_pid_running(self):
        """Test that we read the information from the core correctly even if we
        have a running process with the same PID"""
        new_core = self.getBuildArtifact("linux-x86_64_not_crashed-pid.dmp")
        self.do_change_pid_in_minidump("linux-x86_64_not_crashed.dmp",
                                       new_core,
                                       self._linux_x86_64_not_crashed_pid_offset,
                                       str(self._linux_x86_64_not_crashed_pid),
                                       str(os.getpid()))
        self.do_test_deeper_stack("linux-x86_64_not_crashed", new_core, os.getpid())

    def test_two_cores_same_pid(self):
        """Test that we handle the situation if we have two core files with the same PID """
        new_core = self.getBuildArtifact("linux-x86_64_not_crashed-pid.dmp")
        self.do_change_pid_in_minidump("linux-x86_64_not_crashed.dmp",
                                       new_core,
                                       self._linux_x86_64_not_crashed_pid_offset,
                                       str(self._linux_x86_64_not_crashed_pid),
                                       str(self._linux_x86_64_pid))
        self.do_test_deeper_stack("linux-x86_64_not_crashed",
                                  new_core, self._linux_x86_64_pid)
        self.test_stack_info_in_minidump()

    def test_local_variables_in_minidump(self):
        """Test that we can examine local variables in a Minidump."""
        # Launch with the Minidump, and inspect a local variable.
        # target create linux-x86_64_not_crashed -c linux-x86_64_not_crashed.dmp
        self.target = self.dbg.CreateTarget("linux-x86_64_not_crashed")
        self.process = self.target.LoadCore("linux-x86_64_not_crashed.dmp")
        self.check_state()
        thread = self.process.GetThreadAtIndex(0)
        frame = thread.GetFrameAtIndex(1)
        value = frame.EvaluateExpression('x')
        self.assertEqual(value.GetValueAsSigned(), 3)
