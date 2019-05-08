//===-- SBMemoryRegionInfoList.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBMemoryRegionInfoList.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBMemoryRegionInfo.h"
#include "lldb/API/SBStream.h"
#include "lldb/Target/MemoryRegionInfo.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

class MemoryRegionInfoListImpl {
public:
  MemoryRegionInfoListImpl() : m_regions() {}

  MemoryRegionInfoListImpl(const MemoryRegionInfoListImpl &rhs)
      : m_regions(rhs.m_regions) {}

  MemoryRegionInfoListImpl &operator=(const MemoryRegionInfoListImpl &rhs) {
    if (this == &rhs)
      return *this;
    m_regions = rhs.m_regions;
    return *this;
  }

  size_t GetSize() const { return m_regions.size(); }

  void Reserve(size_t capacity) { return m_regions.reserve(capacity); }

  void Append(const MemoryRegionInfo &sb_region) {
    m_regions.push_back(sb_region);
  }

  void Append(const MemoryRegionInfoListImpl &list) {
    Reserve(GetSize() + list.GetSize());

    for (const auto &val : list.m_regions)
      Append(val);
  }

  void Clear() { m_regions.clear(); }

  bool GetMemoryRegionInfoAtIndex(size_t index,
                                  MemoryRegionInfo &region_info) {
    if (index >= GetSize())
      return false;
    region_info = m_regions[index];
    return true;
  }

  MemoryRegionInfos &Ref() { return m_regions; }

  const MemoryRegionInfos &Ref() const { return m_regions; }

private:
  MemoryRegionInfos m_regions;
};

MemoryRegionInfos &SBMemoryRegionInfoList::ref() { return m_opaque_up->Ref(); }

const MemoryRegionInfos &SBMemoryRegionInfoList::ref() const {
  return m_opaque_up->Ref();
}

SBMemoryRegionInfoList::SBMemoryRegionInfoList()
    : m_opaque_up(new MemoryRegionInfoListImpl()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBMemoryRegionInfoList);
}

SBMemoryRegionInfoList::SBMemoryRegionInfoList(
    const SBMemoryRegionInfoList &rhs)
    : m_opaque_up(new MemoryRegionInfoListImpl(*rhs.m_opaque_up)) {
  LLDB_RECORD_CONSTRUCTOR(SBMemoryRegionInfoList,
                          (const lldb::SBMemoryRegionInfoList &), rhs);
}

SBMemoryRegionInfoList::~SBMemoryRegionInfoList() {}

const SBMemoryRegionInfoList &SBMemoryRegionInfoList::
operator=(const SBMemoryRegionInfoList &rhs) {
  LLDB_RECORD_METHOD(
      const lldb::SBMemoryRegionInfoList &,
      SBMemoryRegionInfoList, operator=,(const lldb::SBMemoryRegionInfoList &),
      rhs);

  if (this != &rhs) {
    *m_opaque_up = *rhs.m_opaque_up;
  }
  return LLDB_RECORD_RESULT(*this);
}

uint32_t SBMemoryRegionInfoList::GetSize() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBMemoryRegionInfoList, GetSize);

  return m_opaque_up->GetSize();
}

bool SBMemoryRegionInfoList::GetMemoryRegionAtIndex(
    uint32_t idx, SBMemoryRegionInfo &region_info) {
  LLDB_RECORD_METHOD(bool, SBMemoryRegionInfoList, GetMemoryRegionAtIndex,
                     (uint32_t, lldb::SBMemoryRegionInfo &), idx, region_info);

  return m_opaque_up->GetMemoryRegionInfoAtIndex(idx, region_info.ref());
}

void SBMemoryRegionInfoList::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBMemoryRegionInfoList, Clear);

  m_opaque_up->Clear();
}

void SBMemoryRegionInfoList::Append(SBMemoryRegionInfo &sb_region) {
  LLDB_RECORD_METHOD(void, SBMemoryRegionInfoList, Append,
                     (lldb::SBMemoryRegionInfo &), sb_region);

  m_opaque_up->Append(sb_region.ref());
}

void SBMemoryRegionInfoList::Append(SBMemoryRegionInfoList &sb_region_list) {
  LLDB_RECORD_METHOD(void, SBMemoryRegionInfoList, Append,
                     (lldb::SBMemoryRegionInfoList &), sb_region_list);

  m_opaque_up->Append(*sb_region_list);
}

const MemoryRegionInfoListImpl *SBMemoryRegionInfoList::operator->() const {
  return m_opaque_up.get();
}

const MemoryRegionInfoListImpl &SBMemoryRegionInfoList::operator*() const {
  assert(m_opaque_up.get());
  return *m_opaque_up;
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBMemoryRegionInfoList>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBMemoryRegionInfoList, ());
  LLDB_REGISTER_CONSTRUCTOR(SBMemoryRegionInfoList,
                            (const lldb::SBMemoryRegionInfoList &));
  LLDB_REGISTER_METHOD(
      const lldb::SBMemoryRegionInfoList &,
      SBMemoryRegionInfoList, operator=,(
                                  const lldb::SBMemoryRegionInfoList &));
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBMemoryRegionInfoList, GetSize, ());
  LLDB_REGISTER_METHOD(bool, SBMemoryRegionInfoList, GetMemoryRegionAtIndex,
                       (uint32_t, lldb::SBMemoryRegionInfo &));
  LLDB_REGISTER_METHOD(void, SBMemoryRegionInfoList, Clear, ());
  LLDB_REGISTER_METHOD(void, SBMemoryRegionInfoList, Append,
                       (lldb::SBMemoryRegionInfo &));
  LLDB_REGISTER_METHOD(void, SBMemoryRegionInfoList, Append,
                       (lldb::SBMemoryRegionInfoList &));
}

}
}
