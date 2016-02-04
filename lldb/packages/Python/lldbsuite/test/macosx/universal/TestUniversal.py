"""Test aspects of lldb commands on universal binaries."""

from __future__ import print_function



import unittest2
import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class UniversalTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    @add_test_categories(['pyapi'])
    @skipUnlessDarwin
    @unittest2.skipUnless(hasattr(os, "uname") and os.uname()[4] in ['i386', 'x86_64'],
            "requires i386 or x86_64")
    def test_sbdebugger_create_target_with_file_and_target_triple(self):
        """Test the SBDebugger.CreateTargetWithFileAndTargetTriple() API."""
        # Invoke the default build rule.
        self.build()

        # Note that "testit" is a universal binary.
        exe = os.path.join(os.getcwd(), "testit")

        # Create a target by the debugger.
        target = self.dbg.CreateTargetWithFileAndTargetTriple(exe, "i386-apple-macosx")
        self.assertTrue(target, VALID_TARGET)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

    @skipUnlessDarwin
    @unittest2.skipUnless(hasattr(os, "uname") and os.uname()[4] in ['i386', 'x86_64'],
            "requires i386 or x86_64")
    def test_process_launch_for_universal(self):
        """Test process launch of a universal binary."""
        from lldbsuite.test.lldbutil import print_registers

        # Invoke the default build rule.
        self.build()

        # Note that "testit" is a universal binary.
        exe = os.path.join(os.getcwd(), "testit")

        # By default, x86_64 is assumed if no architecture is specified.
        self.expect("file " + exe, CURRENT_EXECUTABLE_SET,
            startstr = "Current executable set to ",
            substrs = ["testit' (x86_64)."])

        # Break inside the main.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        # We should be able to launch the x86_64 executable.
        self.runCmd("run", RUN_SUCCEEDED)

        # Check whether we have a 64-bit process launched.
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        self.assertTrue(target and process and
                        self.invoke(process, 'GetAddressByteSize') == 8,
                        "64-bit process launched")

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        registers = print_registers(frame, string_buffer=True)
        self.expect(registers, exe=False,
            substrs = ['Name: rax'])

        self.runCmd("continue")

        # Now specify i386 as the architecture for "testit".
        self.expect("file -a i386 " + exe, CURRENT_EXECUTABLE_SET,
            startstr = "Current executable set to ",
            substrs = ["testit' (i386)."])

        # Break inside the main.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        # We should be able to launch the i386 executable as well.
        self.runCmd("run", RUN_SUCCEEDED)

        # Check whether we have a 32-bit process launched.
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        self.assertTrue(target and process,
                        "32-bit process launched")

        pointerSize = self.invoke(process, 'GetAddressByteSize')
        self.assertTrue(pointerSize == 4,
                        "AddressByteSize of 32-bit process should be 4, got %d instead." % pointerSize)

        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        registers = print_registers(frame, string_buffer=True)
        self.expect(registers, exe=False,
            substrs = ['Name: eax'])

        self.runCmd("continue")
