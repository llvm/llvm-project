"""
Test many basic expression commands and SBFrame.EvaluateExpression() API.

Test cases:

o test_many_expr_commands:
  Test many basic expression commands.
o test_evaluate_expression_python:
  Use Python APIs (SBFrame.EvaluateExpression()) to evaluate expressions.
o test_expr_commands_can_handle_quotes:
  Throw some expression commands with quotes at lldb.
"""

from __future__ import print_function



import unittest2

import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class BasicExprCommandsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.cpp',
                                '// Please test many expressions while stopped at this line:')

        # Disable confirmation prompt to avoid infinite wait
        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(lambda: self.runCmd("settings clear auto-confirm"))


    def build_and_run(self):
        """These basic expression commands should work as expected."""
        self.build()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=False)

        self.runCmd("run", RUN_SUCCEEDED)

    @unittest2.expectedFailure("llvm.org/pr17135 <rdar://problem/14874559> APFloat::toString does not identify the correct (i.e. least) precision.")
    def test_floating_point_expr_commands(self):
        self.build_and_run()

        self.expect("expression 2.234f",
            patterns = ["\(float\) \$.* = 2\.234"])
        # (float) $2 = 2.234

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr21765")
    def test_many_expr_commands(self):
        self.build_and_run()

        self.expect("expression 2",
            patterns = ["\(int\) \$.* = 2"])
        # (int) $0 = 1

        self.expect("expression 2ull",
            patterns = ["\(unsigned long long\) \$.* = 2"])
        # (unsigned long long) $1 = 2

        self.expect("expression 0.5f",
            patterns = ["\(float\) \$.* = 0\.5"])
        # (float) $2 = 0.5

        self.expect("expression 2.234",
            patterns = ["\(double\) \$.* = 2\.234"])
        # (double) $3 = 2.234

        self.expect("expression 2+3",
            patterns = ["\(int\) \$.* = 5"])
        # (int) $4 = 5

        self.expect("expression argc",
            patterns = ["\(int\) \$.* = 1"])
        # (int) $5 = 1

        self.expect("expression argc + 22",
            patterns = ["\(int\) \$.* = 23"])
        # (int) $6 = 23

        self.expect("expression argv",
            patterns = ["\(const char \*\*\) \$.* = 0x"])
        # (const char *) $7 = ...

        self.expect("expression argv[0]",
            substrs = ["(const char *)", 
                       "a.out"])
        # (const char *) $8 = 0x... "/Volumes/data/lldb/svn/trunk/test/expression_command/test/a.out"

    @add_test_categories(['pyapi'])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr21765")
    def test_evaluate_expression_python(self):
        """Test SBFrame.EvaluateExpression() API for evaluating an expression."""
        self.build()

        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint.
        filespec = lldb.SBFileSpec("main.cpp", False)
        breakpoint = target.BreakpointCreateByLocation(filespec, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Verify the breakpoint just created.
        self.expect(str(breakpoint), BREAKPOINT_CREATED, exe=False,
            substrs = ['main.cpp',
                       str(self.line)])

        # Launch the process, and do not stop at the entry point.
        # Pass 'X Y Z' as the args, which makes argc == 4.
        process = target.LaunchSimple (['X', 'Y', 'Z'], None, self.get_process_working_directory())

        if not process:
            self.fail("SBTarget.LaunchProcess() failed")

        if process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.state_type_to_str(process.GetState()))

        thread = lldbutil.get_one_thread_stopped_at_breakpoint(process, breakpoint)
        self.assertIsNotNone(thread, "Expected one thread to be stopped at the breakpoint")

        # The filename of frame #0 should be 'main.cpp' and function is main.
        self.expect(lldbutil.get_filenames(thread)[0],
                    "Break correctly at main.cpp", exe=False,
            startstr = "main.cpp")
        self.expect(lldbutil.get_function_names(thread)[0],
                    "Break correctly at main()", exe=False,
            startstr = "main")

        # We should be stopped on the breakpoint with a hit count of 1.
        self.assertTrue(breakpoint.GetHitCount() == 1, BREAKPOINT_HIT_ONCE)

        #
        # Use Python API to evaluate expressions while stopped in a stack frame.
        #
        frame = thread.GetFrameAtIndex(0)

        val = frame.EvaluateExpression("2.234")
        self.expect(val.GetValue(), "2.345 evaluated correctly", exe=False,
            startstr = "2.234")
        self.expect(val.GetTypeName(), "2.345 evaluated correctly", exe=False,
            startstr = "double")
        self.DebugSBValue(val)

        val = frame.EvaluateExpression("argc")
        self.expect(val.GetValue(), "Argc evaluated correctly", exe=False,
            startstr = "4")
        self.DebugSBValue(val)

        val = frame.EvaluateExpression("*argv[1]")
        self.expect(val.GetValue(), "Argv[1] evaluated correctly", exe=False,
            startstr = "'X'")
        self.DebugSBValue(val)

        val = frame.EvaluateExpression("*argv[2]")
        self.expect(val.GetValue(), "Argv[2] evaluated correctly", exe=False,
            startstr = "'Y'")
        self.DebugSBValue(val)

        val = frame.EvaluateExpression("*argv[3]")
        self.expect(val.GetValue(), "Argv[3] evaluated correctly", exe=False,
            startstr = "'Z'")
        self.DebugSBValue(val)

        callee_break = target.BreakpointCreateByName ("a_function_to_call", None)
        self.assertTrue(callee_break.GetNumLocations() > 0)

        # Make sure ignoring breakpoints works from the command line:
        self.expect("expression -i true -- a_function_to_call()",
                    substrs = ['(int) $', ' 1'])
        self.assertTrue (callee_break.GetHitCount() == 1)

        # Now try ignoring breakpoints using the SB API's:
        options = lldb.SBExpressionOptions()
        options.SetIgnoreBreakpoints(True)
        value = frame.EvaluateExpression('a_function_to_call()', options)
        self.assertTrue (value.IsValid())
        self.assertTrue (value.GetValueAsSigned(0) == 2)
        self.assertTrue (callee_break.GetHitCount() == 2)

    # rdar://problem/8686536
    # CommandInterpreter::HandleCommand is stripping \'s from input for WantsRawCommand commands
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr21765")
    def test_expr_commands_can_handle_quotes(self):
        """Throw some expression commands with quotes at lldb."""
        self.build()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(self, "main.cpp", self.line, num_expected_locations=1,loc_exact=False)

        self.runCmd("run", RUN_SUCCEEDED)

        # runCmd: expression 'a'
        # output: (char) $0 = 'a'
        self.expect("expression 'a'",
            substrs = ['(char) $',
                       "'a'"])

        # runCmd: expression (int) printf ("\n\n\tHello there!\n")
        # output: (int) $1 = 16
        self.expect(r'''expression (int) printf ("\n\n\tHello there!\n")''',
            substrs = ['(int) $',
                       '16'])

        # runCmd: expression (int) printf("\t\x68\n")
        # output: (int) $2 = 3
        self.expect(r'''expression (int) printf("\t\x68\n")''',
            substrs = ['(int) $',
                       '3'])

        # runCmd: expression (int) printf("\"\n")
        # output: (int) $3 = 2
        self.expect(r'''expression (int) printf("\"\n")''',
            substrs = ['(int) $',
                       '2'])

        # runCmd: expression (int) printf("'\n")
        # output: (int) $4 = 2
        self.expect(r'''expression (int) printf("'\n")''',
            substrs = ['(int) $',
                       '2'])

        # runCmd: command alias print_hi expression (int) printf ("\n\tHi!\n")
        # output: 
        self.runCmd(r'''command alias print_hi expression (int) printf ("\n\tHi!\n")''')
        # This fails currently.
        self.expect('print_hi',
            substrs = ['(int) $',
                       '6'])
