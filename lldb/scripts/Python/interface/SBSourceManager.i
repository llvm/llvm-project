//===-- SWIG Interface for SBSourceManager ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a central authority for displaying source code.

For example (from test/source-manager/TestSourceManager.py),

        # Create the filespec for 'main.c'.
        filespec = lldb.SBFileSpec('main.c', False)
        source_mgr = self.dbg.GetSourceManager()
        # Use a string stream as the destination.
        stream = lldb.SBStream()
        source_mgr.DisplaySourceLinesWithLineNumbers(filespec,
                                                     self.line,
                                                     2, # context before
                                                     2, # context after
                                                     '=>', # prefix for current line
                                                     stream)

        #    2    
        #    3    int main(int argc, char const *argv[]) {
        # => 4        printf('Hello world.\\n'); // Set break point at this line.
        #    5        return 0;
        #    6    }
        self.expect(stream.GetData(), 'Source code displayed correctly',
                    exe=False,
            patterns = ['=> %d.*Hello world' % self.line])
") SBSourceManager;
class SBSourceManager
{
public:
    SBSourceManager (const lldb::SBSourceManager &rhs);
    
    ~SBSourceManager();

    size_t
    DisplaySourceLinesWithLineNumbers (const lldb::SBFileSpec &file,
                                       uint32_t line,
                                       uint32_t context_before,
                                       uint32_t context_after,
                                       const char* current_line_cstr,
                                       lldb::SBStream &s);
};

} // namespace lldb
