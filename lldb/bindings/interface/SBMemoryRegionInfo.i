//===-- SWIG Interface for SBMemoryRegionInfo -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

    STRING_EXTENSION(SBMemoryRegionInfo)
};

} // namespace lldb
