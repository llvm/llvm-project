//===-- SWIG Interface for SBMemoryRegionInfo -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"API clients can get information about memory regions in processes."
) SBMemoryRegionInfo;

class SBMemoryRegionInfo
{
public:

    SBMemoryRegionInfo ();

    SBMemoryRegionInfo (const lldb::SBMemoryRegionInfo &rhs);

    ~SBMemoryRegionInfo ();

    void
    Clear();

    lldb::addr_t
    GetRegionBase ();

    lldb::addr_t
    GetRegionEnd ();

    bool
    IsReadable ();

    bool
    IsWritable ();

    bool
    IsExecutable ();

    bool
    IsMapped ();

    const char *
    GetName ();

    bool
    operator == (const lldb::SBMemoryRegionInfo &rhs) const;

    bool
    operator != (const lldb::SBMemoryRegionInfo &rhs) const;

    bool
    GetDescription (lldb::SBStream &description);

};

} // namespace lldb
