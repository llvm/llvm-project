"""
Test some expressions involving STL data types.
"""

from __future__ import print_function


import unittest2
import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class STLTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = 'main.cpp'
        self.line = line_number(
            self.source, '// Set break point at this line.')

    @expectedFailureAll(bugnumber="llvm.org/PR36713")
    def test(self):
        """Test some expressions involving STL data types."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # The following two lines, if uncommented, will enable loggings.
        #self.ci.HandleCommand("log enable -f /tmp/lldb.log lldb default", res)
        # self.assertTrue(res.Succeeded())

        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Stop at 'std::string hello_world ("Hello World!");'.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['main.cpp:%d' % self.line,
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # Now try some expressions....

        self.runCmd(
            'expr for (int i = 0; i < hello_world.length(); ++i) { (void)printf("%c\\n", hello_world[i]); }')

        self.expect('expr associative_array.size()',
                    substrs=[' = 3'])
        self.expect('expr associative_array.count(hello_world)',
                    substrs=[' = 1'])
        self.expect('expr associative_array[hello_world]',
                    substrs=[' = 1'])
        self.expect('expr associative_array["hello"]',
                    substrs=[' = 2'])

    @expectedFailureAll(
        compiler="icc",
        bugnumber="ICC (13.1, 14-beta) do not emit DW_TAG_template_type_parameter.")
    @add_test_categories(['pyapi'])
    def test_SBType_template_aspects(self):
        """Test APIs for getting template arguments from an SBType."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint inside function 'main'.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Get Frame #0.
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)

        # Get the type for variable 'associative_array'.
        associative_array = frame0.FindVariable('associative_array')
        self.DebugSBValue(associative_array)
        self.assertTrue(associative_array, VALID_VARIABLE)
        map_type = associative_array.GetType()
        self.DebugSBType(map_type)
        self.assertTrue(map_type, VALID_TYPE)
        num_template_args = map_type.GetNumberOfTemplateArguments()
        self.assertTrue(num_template_args > 0)

        # We expect the template arguments to contain at least 'string' and
        # 'int'.
        expected_types = {'string': False, 'int': False}
        for i in range(num_template_args):
            t = map_type.GetTemplateArgumentType(i)
            self.DebugSBType(t)
            self.assertTrue(t, VALID_TYPE)
            name = t.GetName()
            if 'string' in name:
                expected_types['string'] = True
            elif 'int' == name:
                expected_types['int'] = True

        # Check that both entries of the dictionary have 'True' as the value.
        self.assertTrue(all(expected_types.values()))
