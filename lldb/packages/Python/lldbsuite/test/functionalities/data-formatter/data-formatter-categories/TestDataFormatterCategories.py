"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class CategoriesDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case (most of these categories do not
        # exist anymore, but we just make sure we delete all of them)
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type category delete Category1', check=False)
            self.runCmd('type category delete Category2', check=False)
            self.runCmd('type category delete NewCategory', check=False)
            self.runCmd("type category delete CircleCategory", check=False)
            self.runCmd("type category delete RectangleStarCategory", check=False)
            self.runCmd("type category delete BaseCategory", check=False)


        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # Add a summary to a new category and check that it works
        self.runCmd("type summary add Rectangle --summary-string \"ARectangle\" -w NewCategory")

        self.expect("frame variable r1 r2 r3", matching=False,
            substrs = ['r1 = ARectangle',
                       'r2 = ARectangle',
                       'r3 = ARectangle'])
        
        self.runCmd("type category enable NewCategory")

        self.expect("frame variable r1 r2 r3", matching=True,
                    substrs = ['r1 = ARectangle',
                               'r2 = ARectangle',
                               'r3 = ARectangle'])
        
        # Disable the category and check that the old stuff is there
        self.runCmd("type category disable NewCategory")

        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = {',
                               'r2 = {',
                               'r3 = {'])

        # Re-enable the category and check that it works
        self.runCmd("type category enable NewCategory")

        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = ARectangle',
                               'r2 = ARectangle',
                               'r3 = ARectangle'])

        # Delete the category and the old stuff should be there
        self.runCmd("type category delete NewCategory")

        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = {',
                               'r2 = {',
                               'r3 = {'])

        # Add summaries to two different categories and check that we can switch
        self.runCmd("type summary add --summary-string \"Width = ${var.w}, Height = ${var.h}\" Rectangle -w Category1")
        self.runCmd("type summary add --python-script \"return 'Area = ' + str( int(valobj.GetChildMemberWithName('w').GetValue()) * int(valobj.GetChildMemberWithName('h').GetValue()) );\" Rectangle -w Category2")

        # check that enable A B is the same as enable B enable A
        self.runCmd("type category enable Category1 Category2")
        
        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = Width = ',
                               'r2 = Width = ',
                               'r3 = Width = '])

        self.runCmd("type category disable Category1")

        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = Area = ',
                               'r2 = Area = ',
                               'r3 = Area = '])

        # switch again

        self.runCmd("type category enable Category1")

        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = Width = ',
                               'r2 = Width = ',
                               'r3 = Width = '])

        # Re-enable the category and show that the preference is persisted
        self.runCmd("type category disable Category2")
        self.runCmd("type category enable Category2")
        
        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = Area = ',
                               'r2 = Area = ',
                               'r3 = Area = '])

        # Now delete the favorite summary
        self.runCmd("type summary delete Rectangle -w Category2")

        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = Width = ',
                               'r2 = Width = ',
                               'r3 = Width = '])

        # Delete the summary from the default category (that does not have it)
        self.runCmd("type summary delete Rectangle", check=False)

        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = Width = ',
                               'r2 = Width = ',
                               'r3 = Width = '])

        # Now add another summary to another category and switch back and forth
        self.runCmd("type category delete Category1 Category2")

        self.runCmd("type summary add Rectangle -w Category1 --summary-string \"Category1\"")
        self.runCmd("type summary add Rectangle -w Category2 --summary-string \"Category2\"")

        self.runCmd("type category enable Category2")
        self.runCmd("type category enable Category1")
        
        self.runCmd("type summary list -w Category1")
        
        self.expect("frame variable r1 r2 r3",
                substrs = ['r1 = Category1',
                           'r2 = Category1',
                           'r3 = Category1'])

        self.runCmd("type category disable Category1")

        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = Category2',
                               'r2 = Category2',
                               'r3 = Category2'])

        # Check that re-enabling an enabled category works
        self.runCmd("type category enable Category1")

        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = Category1',
                               'r2 = Category1',
                               'r3 = Category1'])

        self.runCmd("type category delete Category1")
        self.runCmd("type category delete Category2")

        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = {',
                               'r2 = {',
                               'r3 = {'])

        # Check that multiple summaries can go into one category 
        self.runCmd("type summary add -w Category1 --summary-string \"Width = ${var.w}, Height = ${var.h}\" Rectangle")
        self.runCmd("type summary add -w Category1 --summary-string \"Radius = ${var.r}\" Circle")
        
        self.runCmd("type category enable Category1")

        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = Width = ',
                               'r2 = Width = ',
                               'r3 = Width = '])

        self.expect("frame variable c1 c2 c3",
                    substrs = ['c1 = Radius = ',
                               'c2 = Radius = ',
                               'c3 = Radius = '])

        self.runCmd("type summary delete Circle -w Category1")
            
        self.expect("frame variable c1 c2 c3",
                    substrs = ['c1 = {',
                               'c2 = {',
                               'c3 = {'])

        # Add a regex based summary to a category
        self.runCmd("type summary add -w Category1 --summary-string \"Radius = ${var.r}\" -x Circle")

        self.expect("frame variable r1 r2 r3",
                    substrs = ['r1 = Width = ',
                               'r2 = Width = ',
                               'r3 = Width = '])

        self.expect("frame variable c1 c2 c3",
                    substrs = ['c1 = Radius = ',
                               'c2 = Radius = ',
                               'c3 = Radius = '])

        # Delete it
        self.runCmd("type summary delete Circle -w Category1")

        self.expect("frame variable c1 c2 c3",
                    substrs = ['c1 = {',
                               'c2 = {',
                               'c3 = {'])
        
        # Change a summary inside a category and check that the change is reflected
        self.runCmd("type summary add Circle -w Category1 --summary-string \"summary1\"")

        self.expect("frame variable c1 c2 c3",
                    substrs = ['c1 = summary1',
                               'c2 = summary1',
                               'c3 = summary1'])

        self.runCmd("type summary add Circle -w Category1 --summary-string \"summary2\"")
        
        self.expect("frame variable c1 c2 c3",
                    substrs = ['c1 = summary2',
                               'c2 = summary2',
                               'c3 = summary2'])

        # Check that our order of priority works. Start by clearing categories
        self.runCmd("type category delete Category1")

        self.runCmd("type summary add Shape -w BaseCategory --summary-string \"AShape\"")
        self.runCmd("type category enable BaseCategory")

        self.expect("print (Shape*)&c1",
            substrs = ['AShape'])
        self.expect("print (Shape*)&r1",
            substrs = ['AShape'])
        self.expect("print (Shape*)c_ptr",
            substrs = ['AShape'])
        self.expect("print (Shape*)r_ptr",
            substrs = ['AShape'])

        self.runCmd("type summary add Circle -w CircleCategory --summary-string \"ACircle\"")
        self.runCmd("type summary add Rectangle -w RectangleCategory --summary-string \"ARectangle\"")
        self.runCmd("type category enable CircleCategory")

        self.expect("frame variable c1",
                    substrs = ['ACircle'])
        self.expect("frame variable c_ptr",
                    substrs = ['ACircle'])

        self.runCmd("type summary add \"Rectangle *\" -w RectangleStarCategory --summary-string \"ARectangleStar\"")
        self.runCmd("type category enable RectangleStarCategory")

        self.expect("frame variable c1 r1 c_ptr r_ptr",
                substrs = ['ACircle',
                           'ARectangleStar'])

        self.runCmd("type category enable RectangleCategory")

        self.expect("frame variable c1 r1 c_ptr r_ptr",
                    substrs = ['ACircle',
                               'ACircle',
                               'ARectangle'])

        # Check that abruptly deleting an enabled category does not crash us
        self.runCmd("type category delete RectangleCategory")

        self.expect("frame variable c1 r1 c_ptr r_ptr",
                    substrs = ['ACircle',
                               '(Rectangle) r1 = ', 'w = 5', 'h = 6',
                               'ACircle',
                               'ARectangleStar'])
        
        # check that list commands work
        self.expect("type category list",
                substrs = ['RectangleStarCategory (enabled)'])

        self.expect("type summary list",
                substrs = ['ARectangleStar'])

        # Disable a category and check that it fallsback
        self.runCmd("type category disable CircleCategory")
        
        # check that list commands work
        self.expect("type category list",
                    substrs = ['CircleCategory (disabled'])

        self.expect("frame variable c1 r_ptr",
                    substrs = ['AShape',
                               'ARectangleStar'])

        # check that filters work into categories
        self.runCmd("type filter add Rectangle --child w --category RectangleCategory")
        self.runCmd("type category enable RectangleCategory")
        self.runCmd("type summary add Rectangle --category RectangleCategory --summary-string \" \" -e")
        self.expect('frame variable r2',
            substrs = ['w = 9'])
        self.runCmd("type summary add Rectangle --summary-string \" \" -e")
        self.expect('frame variable r2', matching=False,
                    substrs = ['h = 16'])

        # Now delete all categories
        self.runCmd("type category delete CircleCategory RectangleStarCategory BaseCategory RectangleCategory")

        # check that a deleted category with filter does not blow us up
        self.expect('frame variable r2',
                    substrs = ['w = 9',
                               'h = 16'])

        # and also validate that one can print formatters for a language
        self.expect('type summary list -l c++', substrs=['vector', 'map', 'list', 'string'])
