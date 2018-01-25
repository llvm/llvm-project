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


class LibcxxVectorDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(debug_info="gmodules",
            bugnumber="https://bugs.llvm.org/show_bug.cgi?id=36048")
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        bkpt = self.target().FindBreakpointByID(
            lldbutil.run_break_set_by_source_regexp(
                self, "break here"))

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

        # empty vectors (and storage pointers SHOULD BOTH BE NULL..)
        self.expect("frame variable numbers",
                    substrs=['numbers = size=0'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        # first value added
        self.expect("frame variable numbers",
                    substrs=['numbers = size=1',
                             '[0] = 1',
                             '}'])

        # add some more data
        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect("frame variable numbers",
                    substrs=['numbers = size=4',
                             '[0] = 1',
                             '[1] = 12',
                             '[2] = 123',
                             '[3] = 1234',
                             '}'])

        self.expect("p numbers",
                    substrs=['$', 'size=4',
                             '[0] = 1',
                             '[1] = 12',
                             '[2] = 123',
                             '[3] = 1234',
                             '}'])

        # check access to synthetic children
        self.runCmd(
            "type summary add --summary-string \"item 0 is ${var[0]}\" std::int_vect int_vect")
        self.expect('frame variable numbers',
                    substrs=['item 0 is 1'])

        self.runCmd(
            "type summary add --summary-string \"item 0 is ${svar[0]}\" std::int_vect int_vect")
        self.expect('frame variable numbers',
                    substrs=['item 0 is 1'])
        # move on with synths
        self.runCmd("type summary delete std::int_vect")
        self.runCmd("type summary delete int_vect")

        # add some more data
        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect("frame variable numbers",
                    substrs=['numbers = size=7',
                             '[0] = 1',
                             '[1] = 12',
                             '[2] = 123',
                             '[3] = 1234',
                             '[4] = 12345',
                             '[5] = 123456',
                             '[6] = 1234567',
                             '}'])

        self.expect("p numbers",
                    substrs=['$', 'size=7',
                             '[0] = 1',
                             '[1] = 12',
                             '[2] = 123',
                             '[3] = 1234',
                             '[4] = 12345',
                             '[5] = 123456',
                             '[6] = 1234567',
                             '}'])

        # check access-by-index
        self.expect("frame variable numbers[0]",
                    substrs=['1'])
        self.expect("frame variable numbers[1]",
                    substrs=['12'])
        self.expect("frame variable numbers[2]",
                    substrs=['123'])
        self.expect("frame variable numbers[3]",
                    substrs=['1234'])

        # clear out the vector and see that we do the right thing once again
        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect("frame variable numbers",
                    substrs=['numbers = size=0'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        # first value added
        self.expect("frame variable numbers",
                    substrs=['numbers = size=1',
                             '[0] = 7',
                             '}'])

        # check if we can display strings
        self.expect("frame variable strings",
                    substrs=['goofy',
                             'is',
                             'smart'])

        self.expect("p strings",
                    substrs=['goofy',
                             'is',
                             'smart'])

        # test summaries based on synthetic children
        self.runCmd(
            "type summary add std::string_vect string_vect --summary-string \"vector has ${svar%#} items\" -e")
        self.expect("frame variable strings",
                    substrs=['vector has 3 items',
                             'goofy',
                             'is',
                             'smart'])

        self.expect("p strings",
                    substrs=['vector has 3 items',
                             'goofy',
                             'is',
                             'smart'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect("frame variable strings",
                    substrs=['vector has 4 items'])

        # check access-by-index
        self.expect("frame variable strings[0]",
                    substrs=['goofy'])
        self.expect("frame variable strings[1]",
                    substrs=['is'])

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.expect("frame variable strings",
                    substrs=['vector has 0 items'])
