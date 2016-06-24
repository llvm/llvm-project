//===-- SBMemoryRegionInfoList.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBMemoryRegionInfoList
{
public:

    SBMemoryRegionInfoList ();

    SBMemoryRegionInfoList (const lldb::SBMemoryRegionInfoList &rhs);

    ~SBMemoryRegionInfoList ();

    uint32_t
    GetSize () const;

    bool
    GetMemoryRegionAtIndex (uint32_t idx, SBMemoryRegionInfo &region_info);

    void
    Append (lldb::SBMemoryRegionInfo &region);

    void
    Append (lldb::SBMemoryRegionInfoList &region_list);

    void
    Clear ();
};

} // namespace lldb
