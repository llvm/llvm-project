"""Look up enum type information and check for correct display."""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *


class EnumTypesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    def test(self):
        """Test 'image lookup -t days' and check for correct display and enum value printing."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_to_source_breakpoint(
            self, '// Breakpoint for bitfield', lldb.SBFileSpec("main.c"))

        self.expect("fr var a", DATA_TYPES_DISPLAYED_CORRECTLY,
                    patterns=[' = A$'])
        self.expect("fr var b", DATA_TYPES_DISPLAYED_CORRECTLY,
                    patterns=[' = B$'])
        self.expect("fr var c", DATA_TYPES_DISPLAYED_CORRECTLY,
                    patterns=[' = C$'])
        self.expect("fr var ab", DATA_TYPES_DISPLAYED_CORRECTLY,
                    patterns=[' = AB$'])
        self.expect("fr var ac", DATA_TYPES_DISPLAYED_CORRECTLY,
                    patterns=[' = A | C$'])
        self.expect("fr var all", DATA_TYPES_DISPLAYED_CORRECTLY,
                    patterns=[' = ALL$'])
        # Test that an enum that doesn't match the heuristic we use in
        # ClangASTContext::DumpEnumValue, gets printed as a raw integer.
        self.expect("fr var omega", DATA_TYPES_DISPLAYED_CORRECTLY,
                    patterns=[' = 7$'])
        # Test the behavior in case have a variable of a type considered
        # 'bitfield' by the heuristic, but the value isn't actually fully
        # covered by the enumerators.
        self.expect("p (enum bitfield)nonsense", DATA_TYPES_DISPLAYED_CORRECTLY,
                    patterns=[' = B | C | 0x10$'])

        # Break inside the main.
        bkpt_id = lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True)
        self.runCmd("c", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # Look up information about the 'days' enum type.
        # Check for correct display.
        self.expect("image lookup -t days", DATA_TYPES_DISPLAYED_CORRECTLY,
                    substrs=['enum days {',
                             'Monday',
                             'Tuesday',
                             'Wednesday',
                             'Thursday',
                             'Friday',
                             'Saturday',
                             'Sunday',
                             'kNumDays',
                             '}'])

        enum_values = ['-4',
                       'Monday',
                       'Tuesday',
                       'Wednesday',
                       'Thursday',
                       'Friday',
                       'Saturday',
                       'Sunday',
                       'kNumDays',
                       '5']

        # Make sure a pointer to an anonymous enum type does crash LLDB and displays correctly using
        # frame variable and expression commands
        self.expect(
            'frame variable f.op',
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=[
                'ops *',
                'f.op'],
            patterns=['0x0+$'])
        self.expect(
            'frame variable *f.op',
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=[
                'ops',
                '*f.op',
                '<parent is NULL>'])
        self.expect(
            'expr f.op',
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=[
                'ops *',
                '$'],
            patterns=['0x0+$'])
        self.expect(
            'expr *f.op',
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=['error:'],
            error=True)

        bkpt = self.target().FindBreakpointByID(bkpt_id)
        for enum_value in enum_values:
            self.expect(
                "frame variable day",
                'check for valid enumeration value',
                substrs=[enum_value])
            lldbutil.continue_to_breakpoint(self.process(), bkpt)
