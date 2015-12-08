"""
Test lldb core component: SourceManager.

Test cases:

o test_display_source_python:
  Test display of source using the SBSourceManager API.
o test_modify_source_file_while_debugging:
  Test the caching mechanism of the source manager.
"""

from __future__ import print_function



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

class SourceManagerTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')
        configuration.skip_build_and_cleanup = False

    @add_test_categories(['pyapi'])
    def test_display_source_python(self):
        """Test display of source using the SBSourceManager API."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        #
        # Exercise Python APIs to display source lines.
        #

        # Create the filespec for 'main.c'.
        filespec = lldb.SBFileSpec('main.c', False)
        source_mgr = self.dbg.GetSourceManager()
        # Use a string stream as the destination.
        stream = lldb.SBStream()
        source_mgr.DisplaySourceLinesWithLineNumbers(filespec,
                                                     self.line,
                                                     2, # context before
                                                     2, # context after
                                                     "=>", # prefix for current line
                                                     stream)

        #    2   	
        #    3    int main(int argc, char const *argv[]) {
        # => 4        printf("Hello world.\n"); // Set break point at this line.
        #    5        return 0;
        #    6    }
        self.expect(stream.GetData(), "Source code displayed correctly",
                    exe=False,
            patterns = ['=> %d.*Hello world' % self.line])

        # Boundary condition testings for SBStream().  LLDB should not crash!
        stream.Print(None)
        stream.RedirectToFile(None, True)

    def test_move_and_then_display_source(self):
        """Test that target.source-map settings work by moving main.c to hidden/main.c."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Move main.c to hidden/main.c.
        main_c = "main.c"
        main_c_hidden = os.path.join("hidden", main_c)
        os.rename(main_c, main_c_hidden)

        if self.TraceOn():
            system([["ls"]])
            system([["ls", "hidden"]])

        # Restore main.c after the test.
        self.addTearDownHook(lambda: os.rename(main_c_hidden, main_c))

        # Set target.source-map settings.
        self.runCmd("settings set target.source-map %s %s" % (os.getcwd(), os.path.join(os.getcwd(), "hidden")))
        # And verify that the settings work.
        self.expect("settings show target.source-map",
            substrs = [os.getcwd(), os.path.join(os.getcwd(), "hidden")])

        # Display main() and verify that the source mapping has been kicked in.
        self.expect("source list -n main", SOURCE_DISPLAYED_CORRECTLY,
            substrs = ['Hello world'])

    def test_modify_source_file_while_debugging(self):
        """Modify a source file while debugging the executable."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'main.c:%d' % self.line,
                       'stop reason = breakpoint'])

        # Display some source code.
        self.expect("source list -f main.c -l %d" % self.line, SOURCE_DISPLAYED_CORRECTLY,
            substrs = ['Hello world'])

        # The '-b' option shows the line table locations from the debug information
        # that indicates valid places to set source level breakpoints.

        # The file to display is implicit in this case.
        self.runCmd("source list -l %d -c 3 -b" % self.line)
        output = self.res.GetOutput().splitlines()[0]

        # If the breakpoint set command succeeded, we should expect a positive number
        # of breakpoints for the current line, i.e., self.line.
        import re
        m = re.search('^\[(\d+)\].*// Set break point at this line.', output)
        if not m:
            self.fail("Fail to display source level breakpoints")
        self.assertTrue(int(m.group(1)) > 0)

        # Read the main.c file content.
        with open('main.c', 'r') as f:
            original_content = f.read()
            if self.TraceOn():
                print("original content:", original_content)

        # Modify the in-memory copy of the original source code.
        new_content = original_content.replace('Hello world', 'Hello lldb', 1)

        # This is the function to restore the original content.
        def restore_file():
            #print("os.path.getmtime() before restore:", os.path.getmtime('main.c'))
            time.sleep(1)
            with open('main.c', 'wb') as f:
                f.write(original_content)
            if self.TraceOn():
                with open('main.c', 'r') as f:
                    print("content restored to:", f.read())
            # Touch the file just to be sure.
            os.utime('main.c', None)
            if self.TraceOn():
                print("os.path.getmtime() after restore:", os.path.getmtime('main.c'))



        # Modify the source code file.
        with open('main.c', 'wb') as f:
            time.sleep(1)
            f.write(new_content)
            if self.TraceOn():
                print("new content:", new_content)
                print("os.path.getmtime() after writing new content:", os.path.getmtime('main.c'))
            # Add teardown hook to restore the file to the original content.
            self.addTearDownHook(restore_file)

        # Display the source code again.  We should see the updated line.
        self.expect("source list -f main.c -l %d" % self.line, SOURCE_DISPLAYED_CORRECTLY,
            substrs = ['Hello lldb'])
