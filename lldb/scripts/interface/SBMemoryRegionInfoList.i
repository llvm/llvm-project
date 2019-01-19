//===-- SBMemoryRegionInfoList.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
