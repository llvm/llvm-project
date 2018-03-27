""" Testing debugging of a binary with "mixed" dwarf (with/without fission). """
import os
import lldb
import sys
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestMixedDwarfBinary(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    @no_debug_info_test  # Prevent the genaration of the dwarf version of this test
    @add_test_categories(["dwo"])
    @skipUnlessPlatform(["linux"])
    @skipIf(bugnumber="rdar://38550275")
    def test_mixed_dwarf(self):
        """Test that 'frame variable' works
        for the executable built from two source files compiled
        with/whithout -gsplit-dwarf correspondingly."""

        self.build()
        exe = self.getBuildArtifact("a.out")

        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        main_bp = self.target.BreakpointCreateByName("g", "a.out")
        self.assertTrue(main_bp, VALID_BREAKPOINT)

        self.process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(self.process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.assertTrue(self.process.GetState() == lldb.eStateStopped,
                        STOPPED_DUE_TO_BREAKPOINT)

        frame = self.process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        x = frame.FindVariable("x")
        self.assertTrue(x.IsValid(), "x is not valid")
        y = frame.FindVariable("y")
        self.assertTrue(y.IsValid(), "y is not valid")

