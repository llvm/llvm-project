//===-- SBMemoryRegionInfoList.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBMemoryRegionInfoList_h_
#define LLDB_SBMemoryRegionInfoList_h_

#include "lldb/API/SBDefines.h"

class MemoryRegionInfoListImpl;

namespace lldb {

class LLDB_API SBMemoryRegionInfoList {
public:
  SBMemoryRegionInfoList();

  SBMemoryRegionInfoList(const lldb::SBMemoryRegionInfoList &rhs);

  const SBMemoryRegionInfoList &operator=(const SBMemoryRegionInfoList &rhs);

  ~SBMemoryRegionInfoList();

  uint32_t GetSize() const;

  bool GetMemoryRegionAtIndex(uint32_t idx, SBMemoryRegionInfo &region_info);

  void Append(lldb::SBMemoryRegionInfo &region);

  void Append(lldb::SBMemoryRegionInfoList &region_list);

  void Clear();

protected:
  const MemoryRegionInfoListImpl *operator->() const;

  const MemoryRegionInfoListImpl &operator*() const;

private:
  friend class SBProcess;

  lldb_private::MemoryRegionInfos &ref();

  const lldb_private::MemoryRegionInfos &ref() const;

  std::unique_ptr<MemoryRegionInfoListImpl> m_opaque_up;
};

} // namespace lldb

#endif // LLDB_SBMemoryRegionInfoList_h_
