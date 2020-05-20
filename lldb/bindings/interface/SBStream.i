//===-- SWIG Interface for SBStream -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

namespace lldb {

%feature("docstring",
"Represents a destination for streaming data output to. By default, a string
stream is created.

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
            patterns = ['=> %d.*Hello world' % self.line])") SBStream;
class SBStream
{
public:

    SBStream ();

    ~SBStream ();

    bool
    IsValid() const;

    explicit operator bool() const;

    %feature("docstring", "
    If this stream is not redirected to a file, it will maintain a local
    cache for the stream data which can be accessed using this accessor.") GetData;
    const char *
    GetData ();

    %feature("docstring", "
    If this stream is not redirected to a file, it will maintain a local
    cache for the stream output whose length can be accessed using this
    accessor.") GetSize;
    size_t
    GetSize();

    void
    Print (const char* str);

    void
    RedirectToFile (const char *path, bool append);

    void
    RedirectToFile (lldb::SBFile file);

    void
    RedirectToFile (lldb::FileSP file);

    %extend {
        %feature("autodoc", "DEPRECATED, use RedirectToFile");
        void
        RedirectToFileHandle (lldb::FileSP file, bool transfer_fh_ownership) {
            self->RedirectToFile(file);
        }
    }

    void
    RedirectToFileDescriptor (int fd, bool transfer_fh_ownership);

    %feature("docstring", "
    If the stream is redirected to a file, forget about the file and if
    ownership of the file was transferred to this object, close the file.
    If the stream is backed by a local cache, clear this cache.") Clear;
    void
    Clear ();
};

} // namespace lldb
