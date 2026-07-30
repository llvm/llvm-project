%feature("docstring",
"Represents a file specification that divides the path into a directory and
basename.  The string values of the paths are put into uniqued string pools
for fast comparisons and efficient memory usage.

For example, the following code ::

        lineEntry = context.GetLineEntry()
        self.expect(lineEntry.GetFileSpec().GetDirectory(), 'The line entry should have the correct directory',
                    exe=False,
            substrs = [self.mydir])
        self.expect(lineEntry.GetFileSpec().GetFilename(), 'The line entry should have the correct filename',
                    exe=False,
            substrs = ['main.c'])
        self.assertTrue(lineEntry.GetLine() == self.line,
                        'The line entry's line number should match ')

gets the line entry from the symbol context when a thread is stopped.
It gets the file spec corresponding to the line entry and checks that
the filename and the directory matches what we expect.") lldb::SBFileSpec;
