//===-- SWIG Interface for SBCompileUnit ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a compilation unit, or compiled source file.

SBCompileUnit supports line entry iteration. For example,

    # Now get the SBSymbolContext from this frame.  We want everything. :-)
    context = frame0.GetSymbolContext(lldb.eSymbolContextEverything)
    ...

    compileUnit = context.GetCompileUnit()

    for lineEntry in compileUnit:
        print 'line entry: %s:%d' % (str(lineEntry.GetFileSpec()),
                                    lineEntry.GetLine())
        print 'start addr: %s' % str(lineEntry.GetStartAddress())
        print 'end   addr: %s' % str(lineEntry.GetEndAddress())

produces:

line entry: /Volumes/data/lldb/svn/trunk/test/python_api/symbol-context/main.c:20
start addr: a.out[0x100000d98]
end   addr: a.out[0x100000da3]
line entry: /Volumes/data/lldb/svn/trunk/test/python_api/symbol-context/main.c:21
start addr: a.out[0x100000da3]
end   addr: a.out[0x100000da9]
line entry: /Volumes/data/lldb/svn/trunk/test/python_api/symbol-context/main.c:22
start addr: a.out[0x100000da9]
end   addr: a.out[0x100000db6]
line entry: /Volumes/data/lldb/svn/trunk/test/python_api/symbol-context/main.c:23
start addr: a.out[0x100000db6]
end   addr: a.out[0x100000dbc]
...

See also SBSymbolContext and SBLineEntry"
) SBCompileUnit;
class SBCompileUnit
{
public:

    SBCompileUnit ();

    SBCompileUnit (const lldb::SBCompileUnit &rhs);

    ~SBCompileUnit ();

    bool
    IsValid () const;

    lldb::SBFileSpec
    GetFileSpec () const;

    uint32_t
    GetNumLineEntries () const;

    lldb::SBLineEntry
    GetLineEntryAtIndex (uint32_t idx) const;

    uint32_t
    FindLineEntryIndex (uint32_t start_idx,
                        uint32_t line,
                        lldb::SBFileSpec *inline_file_spec) const;

    uint32_t
    FindLineEntryIndex (uint32_t start_idx,
                        uint32_t line,
                        lldb::SBFileSpec *inline_file_spec,
			bool exact) const;

    SBFileSpec
    GetSupportFileAtIndex (uint32_t idx) const;

    uint32_t
    GetNumSupportFiles () const;

    uint32_t
    FindSupportFileIndex (uint32_t start_idx, const SBFileSpec &sb_file, bool full);

    bool
    GetDescription (lldb::SBStream &description);
    
    bool
    operator == (const lldb::SBCompileUnit &rhs) const;
    
    bool
    operator != (const lldb::SBCompileUnit &rhs) const;
    
    %pythoncode %{
        __swig_getmethods__["file"] = GetFileSpec
        if _newclass: file = property(GetFileSpec, None, doc='''A read only property that returns the same result an lldb object that represents the source file (lldb.SBFileSpec) for the compile unit.''')
        
        __swig_getmethods__["num_line_entries"] = GetNumLineEntries
        if _newclass: num_line_entries = property(GetNumLineEntries, None, doc='''A read only property that returns the number of line entries in a compile unit as an integer.''')
    %}
};

} // namespace lldb
