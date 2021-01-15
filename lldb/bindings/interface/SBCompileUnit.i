//===-- SWIG Interface for SBCompileUnit ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a compilation unit, or compiled source file.

SBCompileUnit supports line entry iteration. For example,::

    # Now get the SBSymbolContext from this frame.  We want everything. :-)
    context = frame0.GetSymbolContext(lldb.eSymbolContextEverything)
    ...

    compileUnit = context.GetCompileUnit()

    for lineEntry in compileUnit:
        print('line entry: %s:%d' % (str(lineEntry.GetFileSpec()),
                                    lineEntry.GetLine()))
        print('start addr: %s' % str(lineEntry.GetStartAddress()))
        print('end   addr: %s' % str(lineEntry.GetEndAddress()))

produces: ::

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

See also :py:class:`SBSymbolContext` and :py:class:`SBLineEntry`"
) SBCompileUnit;
class SBCompileUnit
{
public:

    SBCompileUnit ();

    SBCompileUnit (const lldb::SBCompileUnit &rhs);

    ~SBCompileUnit ();

    bool
    IsValid () const;

    explicit operator bool() const;

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

    %feature("docstring", "
     Get all types matching type_mask from debug info in this
     compile unit.

     @param[in] type_mask
        A bitfield that consists of one or more bits logically OR'ed
        together from the lldb::TypeClass enumeration. This allows
        you to request only structure types, or only class, struct
        and union types. Passing in lldb::eTypeClassAny will return
        all types found in the debug information for this compile
        unit.

     @return
        A list of types in this compile unit that match type_mask") GetTypes;
    lldb::SBTypeList
    GetTypes (uint32_t type_mask = lldb::eTypeClassAny);

     lldb::LanguageType
     GetLanguage ();

    bool
    GetDescription (lldb::SBStream &description);

    bool
    operator == (const lldb::SBCompileUnit &rhs) const;

    bool
    operator != (const lldb::SBCompileUnit &rhs) const;

    STRING_EXTENSION(SBCompileUnit)

#ifdef SWIGPYTHON
    %pythoncode %{
        def __iter__(self):
            '''Iterate over all line entries in a lldb.SBCompileUnit object.'''
            return lldb_iter(self, 'GetNumLineEntries', 'GetLineEntryAtIndex')

        def __len__(self):
            '''Return the number of line entries in a lldb.SBCompileUnit
            object.'''
            return self.GetNumLineEntries()

        file = property(GetFileSpec, None, doc='''A read only property that returns the same result an lldb object that represents the source file (lldb.SBFileSpec) for the compile unit.''')
        num_line_entries = property(GetNumLineEntries, None, doc='''A read only property that returns the number of line entries in a compile unit as an integer.''')
    %}
#endif
};

} // namespace lldb
