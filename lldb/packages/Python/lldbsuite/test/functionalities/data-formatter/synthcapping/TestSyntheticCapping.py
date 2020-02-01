"""
Check for an issue where capping does not work because the Target pointer appears to be changing behind our backs
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SyntheticCappingTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_with_run_command(self):
        """Check for an issue where capping does not work because the Target pointer appears to be changing behind our backs."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        process = self.dbg.GetSelectedTarget().GetProcess()

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

        # set up the synthetic children provider
        self.runCmd("script from fooSynthProvider import *")
        self.runCmd("type synth add -l fooSynthProvider foo")

        # note that the value of fake_a depends on target byte order
        if process.GetByteOrder() == lldb.eByteOrderLittle:
            fake_a_val = 0x02000000
        else:
            fake_a_val = 0x00000100

        # check that the synthetic children work, so we know we are doing the
        # right thing
        self.expect(
            "frame variable f00_1",
            substrs=[
                'a = 1',
                'fake_a = %d' % fake_a_val,
                'r = 34',
            ])

        # check that capping works
        self.runCmd("settings set target.max-children-count 2", check=False)

        self.expect("frame variable f00_1",
                    substrs=[
                        'a = 1',
                        'fake_a = %d' % fake_a_val,
                        '...',
                    ])

        self.expect("frame variable f00_1", matching=False,
                    substrs=['r = 34'])

        self.runCmd("settings set target.max-children-count 256", check=False)

        self.expect("frame variable f00_1", matching=True,
                    substrs=['r = 34'])
