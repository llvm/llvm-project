"""
Test that the Python operating system plugin works correctly
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class PluginPythonOSPlugin(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_python_os_plugin(self):
        """Test that the Python operating system plugin works correctly"""
        self.build()
        self.run_python_os_funcionality()

    def run_python_os_step(self):
        """Test that the Python operating system plugin works correctly when single stepping a virtual thread"""
        self.build()
        self.run_python_os_step()

    def verify_os_thread_registers(self, thread):
        frame = thread.GetFrameAtIndex(0)
        registers = frame.GetRegisters().GetValueAtIndex(0)
        reg_value = thread.GetThreadID() + 1
        for reg in registers:
            self.assertTrue(
                reg.GetValueAsUnsigned() == reg_value,
                "Verify the registers contains the correct value")
            reg_value = reg_value + 1

    def run_python_os_funcionality(self):
        """Test that the Python operating system plugin works correctly"""

        # Set debugger into synchronous mode
        self.dbg.SetAsync(False)

        # Create a target by the debugger.
        cwd = os.getcwd()
        exe = os.path.join(cwd, "a.out")
        python_os_plugin_path = os.path.join(cwd, "operating_system.py")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set breakpoints inside and outside methods that take pointers to the
        # containing struct.
        lldbutil.run_break_set_by_source_regexp(self, "// Set breakpoint here")

        # Register our shared libraries for remote targets so they get
        # automatically uploaded
        arguments = None
        environment = None

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            arguments, environment, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Make sure there are no OS plug-in created thread when we first stop
        # at our breakpoint in main
        thread = process.GetThreadByID(0x111111111)
        self.assertFalse(
            thread.IsValid(),
            "Make sure there is no thread 0x111111111 before we load the python OS plug-in")
        thread = process.GetThreadByID(0x222222222)
        self.assertFalse(
            thread.IsValid(),
            "Make sure there is no thread 0x222222222 before we load the python OS plug-in")
        thread = process.GetThreadByID(0x333333333)
        self.assertFalse(
            thread.IsValid(),
            "Make sure there is no thread 0x333333333 before we load the python OS plug-in")

        # Now load the python OS plug-in which should update the thread list and we should have
        # OS plug-in created threads with the IDs: 0x111111111, 0x222222222,
        # 0x333333333
        command = "settings set target.process.python-os-plugin-path '%s'" % python_os_plugin_path
        self.dbg.HandleCommand(command)

        # Verify our OS plug-in threads showed up
        thread = process.GetThreadByID(0x111111111)
        self.assertTrue(
            thread.IsValid(),
            "Make sure there is a thread 0x111111111 after we load the python OS plug-in")
        self.verify_os_thread_registers(thread)
        thread = process.GetThreadByID(0x222222222)
        self.assertTrue(
            thread.IsValid(),
            "Make sure there is a thread 0x222222222 after we load the python OS plug-in")
        self.verify_os_thread_registers(thread)
        thread = process.GetThreadByID(0x333333333)
        self.assertTrue(
            thread.IsValid(),
            "Make sure there is a thread 0x333333333 after we load the python OS plug-in")
        self.verify_os_thread_registers(thread)

        # Now clear the OS plug-in path to make the OS plug-in created threads
        # dissappear
        self.dbg.HandleCommand(
            "settings clear target.process.python-os-plugin-path")

        # Verify the threads are gone after unloading the python OS plug-in
        thread = process.GetThreadByID(0x111111111)
        self.assertFalse(
            thread.IsValid(),
            "Make sure there is no thread 0x111111111 after we unload the python OS plug-in")
        thread = process.GetThreadByID(0x222222222)
        self.assertFalse(
            thread.IsValid(),
            "Make sure there is no thread 0x222222222 after we unload the python OS plug-in")
        thread = process.GetThreadByID(0x333333333)
        self.assertFalse(
            thread.IsValid(),
            "Make sure there is no thread 0x333333333 after we unload the python OS plug-in")

    def run_python_os_step(self):
        """Test that the Python operating system plugin works correctly and allows single stepping of a virtual thread that is backed by a real thread"""

        # Set debugger into synchronous mode
        self.dbg.SetAsync(False)

        # Create a target by the debugger.
        cwd = os.getcwd()
        exe = os.path.join(cwd, "a.out")
        python_os_plugin_path = os.path.join(cwd, "operating_system2.py")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set breakpoints inside and outside methods that take pointers to the
        # containing struct.
        lldbutil.run_break_set_by_source_regexp(self, "// Set breakpoint here")

        # Register our shared libraries for remote targets so they get
        # automatically uploaded
        arguments = None
        environment = None

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            arguments, environment, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Make sure there are no OS plug-in created thread when we first stop
        # at our breakpoint in main
        thread = process.GetThreadByID(0x111111111)
        self.assertFalse(
            thread.IsValid(),
            "Make sure there is no thread 0x111111111 before we load the python OS plug-in")

        # Now load the python OS plug-in which should update the thread list and we should have
        # OS plug-in created threads with the IDs: 0x111111111, 0x222222222,
        # 0x333333333
        command = "settings set target.process.python-os-plugin-path '%s'" % python_os_plugin_path
        self.dbg.HandleCommand(command)

        # Verify our OS plug-in threads showed up
        thread = process.GetThreadByID(0x111111111)
        self.assertTrue(
            thread.IsValid(),
            "Make sure there is a thread 0x111111111 after we load the python OS plug-in")

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(
            frame.IsValid(),
            "Make sure we get a frame from thread 0x111111111")
        line_entry = frame.GetLineEntry()

        self.assertTrue(
            line_entry.GetFileSpec().GetFilename() == 'main.c',
            "Make sure we stopped on line 5 in main.c")
        self.assertTrue(
            line_entry.GetLine() == 5,
            "Make sure we stopped on line 5 in main.c")

        # Now single step thread 0x111111111 and make sure it does what we need
        # it to
        thread.StepOver()

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(
            frame.IsValid(),
            "Make sure we get a frame from thread 0x111111111")
        line_entry = frame.GetLineEntry()

        self.assertTrue(
            line_entry.GetFileSpec().GetFilename() == 'main.c',
            "Make sure we stepped from line 5 to line 6 in main.c")
        self.assertTrue(line_entry.GetLine() == 6,
                        "Make sure we stepped from line 5 to line 6 in main.c")
