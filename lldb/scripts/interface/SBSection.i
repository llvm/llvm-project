//===-- SWIG Interface for SBSection ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents an executable image section.

SBSection supports iteration through its subsection, represented as SBSection
as well.  For example,

    for sec in exe_module:
        if sec.GetName() == '__TEXT':
            print sec
            break
    print INDENT + 'Number of subsections: %d' % sec.GetNumSubSections()
    for subsec in sec:
        print INDENT + repr(subsec)

produces:

[0x0000000100000000-0x0000000100002000) a.out.__TEXT
    Number of subsections: 6
    [0x0000000100001780-0x0000000100001d5c) a.out.__TEXT.__text
    [0x0000000100001d5c-0x0000000100001da4) a.out.__TEXT.__stubs
    [0x0000000100001da4-0x0000000100001e2c) a.out.__TEXT.__stub_helper
    [0x0000000100001e2c-0x0000000100001f10) a.out.__TEXT.__cstring
    [0x0000000100001f10-0x0000000100001f68) a.out.__TEXT.__unwind_info
    [0x0000000100001f68-0x0000000100001ff8) a.out.__TEXT.__eh_frame

See also SBModule."
) SBSection;

class SBSection
{
public:

    SBSection ();

    SBSection (const lldb::SBSection &rhs);

    ~SBSection ();

    bool
    IsValid () const;

    const char *
    GetName ();

    lldb::SBSection
    GetParent();

    lldb::SBSection
    FindSubSection (const char *sect_name);

    size_t
    GetNumSubSections ();

    lldb::SBSection
    GetSubSectionAtIndex (size_t idx);

    lldb::addr_t
    GetFileAddress ();

    lldb::addr_t
    GetLoadAddress (lldb::SBTarget &target);
    
    lldb::addr_t
    GetByteSize ();

    uint64_t
    GetFileOffset ();

    uint64_t
    GetFileByteSize ();
    
    lldb::SBData
    GetSectionData ();

    lldb::SBData
    GetSectionData (uint64_t offset,
                    uint64_t size);

    SectionType
    GetSectionType ();

    uint32_t
    GetPermissions() const;

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Return the size of a target's byte represented by this section
    /// in numbers of host bytes. Note that certain architectures have
    /// varying minimum addressable unit (i.e. byte) size for their 
    /// CODE or DATA buses.
    ///
    /// @return
    ///     The number of host (8-bit) bytes needed to hold a target byte
    //------------------------------------------------------------------
    ") GetTargetByteSize;
    uint32_t
    GetTargetByteSize ();

    bool
    GetDescription (lldb::SBStream &description);
    
    bool
    operator == (const lldb::SBSection &rhs);

    bool
    operator != (const lldb::SBSection &rhs);
     
    %pythoncode %{
        def get_addr(self):
            return SBAddress(self, 0)

        __swig_getmethods__["name"] = GetName
        if _newclass: name = property(GetName, None, doc='''A read only property that returns the name of this section as a string.''')

        __swig_getmethods__["addr"] = get_addr
        if _newclass: addr = property(get_addr, None, doc='''A read only property that returns an lldb object that represents the start address (lldb.SBAddress) for this section.''')

        __swig_getmethods__["file_addr"] = GetFileAddress
        if _newclass: file_addr = property(GetFileAddress, None, doc='''A read only property that returns an integer that represents the starting "file" address for this section, or the address of the section in the object file in which it is defined.''')

        __swig_getmethods__["size"] = GetByteSize
        if _newclass: size = property(GetByteSize, None, doc='''A read only property that returns the size in bytes of this section as an integer.''')

        __swig_getmethods__["file_offset"] = GetFileOffset
        if _newclass: file_offset = property(GetFileOffset, None, doc='''A read only property that returns the file offset in bytes of this section as an integer.''')

        __swig_getmethods__["file_size"] = GetFileByteSize
        if _newclass: file_size = property(GetFileByteSize, None, doc='''A read only property that returns the file size in bytes of this section as an integer.''')

        __swig_getmethods__["data"] = GetSectionData
        if _newclass: data = property(GetSectionData, None, doc='''A read only property that returns an lldb object that represents the bytes for this section (lldb.SBData) for this section.''')

        __swig_getmethods__["type"] = GetSectionType
        if _newclass: type = property(GetSectionType, None, doc='''A read only property that returns an lldb enumeration value (see enumerations that start with "lldb.eSectionType") that represents the type of this section (code, data, etc.).''')

        __swig_getmethods__["target_byte_size"] = GetTargetByteSize
        if _newclass: target_byte_size = property(GetTargetByteSize, None, doc='''A read only property that returns the size of a target byte represented by this section as a number of host bytes.''')

    %}

private:

    std::unique_ptr<lldb_private::SectionImpl> m_opaque_ap;
};

} // namespace lldb
