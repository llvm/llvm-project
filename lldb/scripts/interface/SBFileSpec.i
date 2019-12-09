//===-- SWIG Interface for SBFileSpec ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a file specification that divides the path into a directory and
basename.  The string values of the paths are put into uniqued string pools
for fast comparisons and efficient memory usage.

For example, the following code

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
the filename and the directory matches what we expect.") SBFileSpec;
class SBFileSpec
{
public:
    SBFileSpec ();

    SBFileSpec (const lldb::SBFileSpec &rhs);

    SBFileSpec (const char *path);// Deprecated, use SBFileSpec (const char *path, bool resolve)

    SBFileSpec (const char *path, bool resolve);

    ~SBFileSpec ();

    bool operator==(const SBFileSpec &rhs) const;

    bool operator!=(const SBFileSpec &rhs) const;

    bool
    IsValid() const;

    explicit operator bool() const;

    bool
    Exists () const;

    bool
    ResolveExecutableLocation ();

    const char *
    GetFilename() const;

    const char *
    GetDirectory() const;

    void
    SetFilename(const char *filename);

    void
    SetDirectory(const char *directory);

    uint32_t
    GetPath (char *dst_path, size_t dst_len) const;

    static int
    ResolvePath (const char *src_path, char *dst_path, size_t dst_len);

    bool
    GetDescription (lldb::SBStream &description) const;

    void
    AppendPathComponent (const char *file_or_directory);

#ifdef SWIGPYTHON
    %pythoncode %{
        def __get_fullpath__(self):
            spec_dir = self.GetDirectory()
            spec_file = self.GetFilename()
            if spec_dir and spec_file:
                return '%s/%s' % (spec_dir, spec_file)
            elif spec_dir:
                return spec_dir
            elif spec_file:
                return spec_file
            return None

        fullpath = property(__get_fullpath__, None, doc='''A read only property that returns the fullpath as a python string.''')
        basename = property(GetFilename, None, doc='''A read only property that returns the path basename as a python string.''')
        dirname = property(GetDirectory, None, doc='''A read only property that returns the path directory name as a python string.''')
        exists = property(Exists, None, doc='''A read only property that returns a boolean value that indicates if the file exists.''')
    %}
#endif

};

} // namespace lldb
