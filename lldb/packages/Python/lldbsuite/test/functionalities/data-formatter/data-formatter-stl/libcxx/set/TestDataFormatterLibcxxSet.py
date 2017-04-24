"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxSetDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        ns = 'ndk' if lldbplatformutil.target_is_android() else ''
        self.namespace = 'std::__' + ns + '1'

    def getVariableType(self, name):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid())
        return var.GetType().GetCanonicalType().GetName()

    @add_test_categories(["libc++"])
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        bkpt = self.target().FindBreakpointByID(
            lldbutil.run_break_set_by_source_regexp(self, "Set break point at this line."))

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd(
                "settings set target.max-children-count 256",
                check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        ii_type = self.getVariableType("ii")
        self.assertTrue(ii_type.startswith(self.namespace + "::set"),
                        "Type: " + ii_type)

        self.expect("frame variable ii", substrs=["size=0", "{}"])
        lldbutil.continue_to_breakpoint(self.process(), bkpt)
        self.expect(
            "frame variable ii",
            substrs=["size=6",
                     "[0] = 0",
                     "[1] = 1",
                     "[2] = 2",
                     "[3] = 3",
                     "[4] = 4",
                     "[5] = 5"])
        lldbutil.continue_to_breakpoint(self.process(), bkpt)
        self.expect(
            "frame variable ii",
            substrs=["size=7",
                     "[2] = 2",
                     "[3] = 3",
                     "[6] = 6"])
        self.expect("frame variable ii[2]", substrs=[" = 2"])
        self.expect(
            "p ii",
            substrs=[
                "size=7",
                "[2] = 2",
                "[3] = 3",
                "[6] = 6"])
        lldbutil.continue_to_breakpoint(self.process(), bkpt)
        self.expect("frame variable ii", substrs=["size=0", "{}"])
        lldbutil.continue_to_breakpoint(self.process(), bkpt)
        self.expect("frame variable ii", substrs=["size=0", "{}"])

        ss_type = self.getVariableType("ss")
        self.assertTrue(ii_type.startswith(self.namespace + "::set"),
                        "Type: " + ss_type)

        self.expect("frame variable ss", substrs=["size=0", "{}"])
        lldbutil.continue_to_breakpoint(self.process(), bkpt)
        self.expect(
            "frame variable ss",
            substrs=["size=2",
                     '[0] = "a"',
                     '[1] = "a very long string is right here"'])
        lldbutil.continue_to_breakpoint(self.process(), bkpt)
        self.expect(
            "frame variable ss",
            substrs=["size=4",
                     '[2] = "b"',
                     '[3] = "c"',
                     '[0] = "a"',
                     '[1] = "a very long string is right here"'])
        self.expect(
            "p ss",
            substrs=["size=4",
                     '[2] = "b"',
                     '[3] = "c"',
                     '[0] = "a"',
                     '[1] = "a very long string is right here"'])
        self.expect("frame variable ss[2]", substrs=[' = "b"'])
        lldbutil.continue_to_breakpoint(self.process(), bkpt)
        self.expect(
            "frame variable ss",
            substrs=["size=3",
                     '[0] = "a"',
                     '[1] = "a very long string is right here"',
                     '[2] = "c"'])
