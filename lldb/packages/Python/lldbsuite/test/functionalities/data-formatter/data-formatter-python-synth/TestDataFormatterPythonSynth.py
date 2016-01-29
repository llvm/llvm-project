"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

class PythonSynthDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfFreeBSD # llvm.org/pr20545 bogus output confuses buildbot parser
    def test_with_run_command(self):
        """Test data formatter commands."""
        self.build()
        self.data_formatter_commands()

    def test_rdar10960550_with_run_command(self):
        """Test data formatter commands."""
        self.build()
        self.rdar10960550_formatter_commands()


    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')
        self.line2 = line_number('main.cpp', '// Set cast break point at this line.')
        self.line3 = line_number('main.cpp', '// Set second cast break point at this line.')

    def data_formatter_commands(self):
        """Test using Python synthetic children provider."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # print the f00_1 variable without a synth
        self.expect("frame variable f00_1",
            substrs = ['a = 0',
                       'b = 1',
                       'r = 33']);

        # now set up the synth
        self.runCmd("script from fooSynthProvider import *")
        self.runCmd("type synth add -l fooSynthProvider foo")
        self.expect("type synthetic list foo", substrs=['fooSynthProvider'])

        # check that we get the two real vars and the fake_a variables
        self.expect("frame variable f00_1",
                    substrs = ['r = 33',
                               'fake_a = 16777216',
                               'a = 0']);

        # check that we do not get the extra vars
        self.expect("frame variable f00_1", matching=False,
                    substrs = ['b = 1']);
        
        # check access to members by name
        self.expect('frame variable f00_1.fake_a',
                substrs = ['16777216'])
        
        # check access to members by index
        self.expect('frame variable f00_1[1]',
                    substrs = ['16777216'])
        
        # put synthetic children in summary in several combinations
        self.runCmd("type summary add --summary-string \"fake_a=${svar.fake_a}\" foo")
        self.expect('frame variable f00_1',
                    substrs = ['fake_a=16777216'])
        self.runCmd("type summary add --summary-string \"fake_a=${svar[1]}\" foo")
        self.expect('frame variable f00_1',
            substrs = ['fake_a=16777216'])
        
        # clear the summary
        self.runCmd("type summary delete foo")

        # check that the caching does not span beyond the stopoint
        self.runCmd("n")

        self.expect("frame variable f00_1",
                    substrs = ['r = 33',
                               'fake_a = 16777216',
                               'a = 1']);

        # check that altering the object also alters fake_a
        self.runCmd("expr f00_1.a = 280")
        self.expect("frame variable f00_1",
                    substrs = ['r = 33',
                               'fake_a = 16777217',
                               'a = 280']);

        # check that expanding a pointer does the right thing
        self.expect("frame variable --ptr-depth 1 f00_ptr",
            substrs = ['r = 45',
                       'fake_a = 218103808',
                       'a = 12'])
        
        # now add a filter.. it should fail
        self.expect("type filter add foo --child b --child j", error=True,
                substrs = ['cannot add'])
        
        # we get the synth again..
        self.expect('frame variable f00_1', matching=False,
            substrs = ['b = 1',
                       'j = 17'])
        self.expect("frame variable --ptr-depth 1 f00_ptr",
                    substrs = ['r = 45',
                               'fake_a = 218103808',
                               'a = 12'])
        
        # now delete the synth and add the filter
        self.runCmd("type synth delete foo")
        self.runCmd("type filter add foo --child b --child j")
        
        self.expect('frame variable f00_1',
                        substrs = ['b = 1',
                                   'j = 17'])
        self.expect("frame variable --ptr-depth 1 f00_ptr", matching=False,
                    substrs = ['r = 45',
                               'fake_a = 218103808',
                               'a = 12'])
        
        # now add the synth and it should fail
        self.expect("type synth add -l fooSynthProvider foo", error=True,
                    substrs = ['cannot add'])
        
        # check the listing
        self.expect('type synth list', matching=False,
                    substrs = ['foo',
                               'Python class fooSynthProvider'])
        self.expect('type filter list', 
                    substrs = ['foo',
                               '.b',
                               '.j'])
        
        # delete the filter, add the synth
        self.runCmd("type filter delete foo")
        self.runCmd("type synth add -l fooSynthProvider foo")
        
        self.expect('frame variable f00_1', matching=False,
                    substrs = ['b = 1',
                               'j = 17'])
        self.expect("frame variable --ptr-depth 1 f00_ptr", 
                    substrs = ['r = 45',
                               'fake_a = 218103808',
                               'a = 12'])

        # check the listing
        self.expect('type synth list',
                    substrs = ['foo',
                               'Python class fooSynthProvider'])
        self.expect('type filter list', matching=False,
                    substrs = ['foo',
                               '.b',
                               '.j'])
        
        # delete the synth and check that we get good output
        self.runCmd("type synth delete foo")
        
        self.expect("frame variable f00_1",
                    substrs = ['a = 280',
                               'b = 1',
                               'j = 17']);

        self.expect("frame variable f00_1", matching=False,
                substrs = ['fake_a = '])

    def rdar10960550_formatter_commands(self):
        """Test that synthetic children persist stoppoints."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        # The second breakpoint is on a multi-line expression, so the comment can't be on the right line...
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line2, num_expected_locations=1, loc_exact=False)
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line3, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("command script import ./ftsp.py --allow-reload")
        self.runCmd("type synth add -l ftsp.ftsp wrapint")

        # we need to check that the VO is properly updated so that the same synthetic children are reused
        # but their values change correctly across stop-points - in order to do this, self.runCmd("next")
        # does not work because it forces a wipe of the stack frame - this is why we are using this more contrived
        # mechanism to achieve our goal of preserving test_cast as a VO
        test_cast = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('test_cast')

        str_cast = str(test_cast)

        if self.TraceOn():
             print(str_cast)

        self.assertTrue(str_cast.find('A') != -1, 'could not find A in output')
        self.assertTrue(str_cast.find('B') != -1, 'could not find B in output')
        self.assertTrue(str_cast.find('C') != -1, 'could not find C in output')
        self.assertTrue(str_cast.find('D') != -1, 'could not find D in output')
        self.assertTrue(str_cast.find("4 = '\\0'") != -1, 'could not find item 4 == 0')

        self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().StepOver()

        str_cast = str(test_cast)

        if self.TraceOn():
             print(str_cast)

        # we detect that all the values of the child objects have changed - but the counter-generated item
        # is still fixed at 0 because it is cached - this would fail if update(self): in ftsp returned False
        # or if synthetic children were not being preserved
        self.assertTrue(str_cast.find('Q') != -1, 'could not find Q in output')
        self.assertTrue(str_cast.find('X') != -1, 'could not find X in output')
        self.assertTrue(str_cast.find('T') != -1, 'could not find T in output')
        self.assertTrue(str_cast.find('F') != -1, 'could not find F in output')
        self.assertTrue(str_cast.find("4 = '\\0'") != -1, 'could not find item 4 == 0')
