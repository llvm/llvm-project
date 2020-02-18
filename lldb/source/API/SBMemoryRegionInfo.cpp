//===-- SBMemoryRegionInfo.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBMemoryRegionInfo.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBStream.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

SBMemoryRegionInfo::SBMemoryRegionInfo() : m_opaque_up(new MemoryRegionInfo()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBMemoryRegionInfo);
}

SBMemoryRegionInfo::SBMemoryRegionInfo(const MemoryRegionInfo *lldb_object_ptr)
    : m_opaque_up(new MemoryRegionInfo()) {
  if (lldb_object_ptr)
    ref() = *lldb_object_ptr;
}

SBMemoryRegionInfo::SBMemoryRegionInfo(const SBMemoryRegionInfo &rhs)
    : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR(SBMemoryRegionInfo,
                          (const lldb::SBMemoryRegionInfo &), rhs);
  m_opaque_up = clone(rhs.m_opaque_up);
}

const SBMemoryRegionInfo &SBMemoryRegionInfo::
operator=(const SBMemoryRegionInfo &rhs) {
  LLDB_RECORD_METHOD(
      const lldb::SBMemoryRegionInfo &,
      SBMemoryRegionInfo, operator=,(const lldb::SBMemoryRegionInfo &), rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return LLDB_RECORD_RESULT(*this);
}

SBMemoryRegionInfo::~SBMemoryRegionInfo() = default;

void SBMemoryRegionInfo::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBMemoryRegionInfo, Clear);

  m_opaque_up->Clear();
}

bool SBMemoryRegionInfo::operator==(const SBMemoryRegionInfo &rhs) const {
  LLDB_RECORD_METHOD_CONST(
      bool, SBMemoryRegionInfo, operator==,(const lldb::SBMemoryRegionInfo &),
      rhs);

  return ref() == rhs.ref();
}

bool SBMemoryRegionInfo::operator!=(const SBMemoryRegionInfo &rhs) const {
  LLDB_RECORD_METHOD_CONST(
      bool, SBMemoryRegionInfo, operator!=,(const lldb::SBMemoryRegionInfo &),
      rhs);

  return ref() != rhs.ref();
}

MemoryRegionInfo &SBMemoryRegionInfo::ref() { return *m_opaque_up; }

const MemoryRegionInfo &SBMemoryRegionInfo::ref() const { return *m_opaque_up; }

lldb::addr_t SBMemoryRegionInfo::GetRegionBase() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::addr_t, SBMemoryRegionInfo, GetRegionBase);

  return m_opaque_up->GetRange().GetRangeBase();
}

lldb::addr_t SBMemoryRegionInfo::GetRegionEnd() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::addr_t, SBMemoryRegionInfo, GetRegionEnd);

  return m_opaque_up->GetRange().GetRangeEnd();
}

bool SBMemoryRegionInfo::IsReadable() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBMemoryRegionInfo, IsReadable);

  return m_opaque_up->GetReadable() == MemoryRegionInfo::eYes;
}

bool SBMemoryRegionInfo::IsWritable() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBMemoryRegionInfo, IsWritable);

  return m_opaque_up->GetWritable() == MemoryRegionInfo::eYes;
}

bool SBMemoryRegionInfo::IsExecutable() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBMemoryRegionInfo, IsExecutable);

  return m_opaque_up->GetExecutable() == MemoryRegionInfo::eYes;
}

bool SBMemoryRegionInfo::IsMapped() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBMemoryRegionInfo, IsMapped);

  return m_opaque_up->GetMapped() == MemoryRegionInfo::eYes;
}

const char *SBMemoryRegionInfo::GetName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBMemoryRegionInfo, GetName);

  return m_opaque_up->GetName().AsCString();
}

bool SBMemoryRegionInfo::GetDescription(SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBMemoryRegionInfo, GetDescription,
                     (lldb::SBStream &), description);

  Stream &strm = description.ref();
  const addr_t load_addr = m_opaque_up->GetRange().base;

  strm.Printf("[0x%16.16" PRIx64 "-0x%16.16" PRIx64 " ", load_addr,
              load_addr + m_opaque_up->GetRange().size);
  strm.Printf(m_opaque_up->GetReadable() ? "R" : "-");
  strm.Printf(m_opaque_up->GetWritable() ? "W" : "-");
  strm.Printf(m_opaque_up->GetExecutable() ? "X" : "-");
  strm.Printf("]");

  return true;
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBMemoryRegionInfo>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBMemoryRegionInfo, ());
  LLDB_REGISTER_CONSTRUCTOR(SBMemoryRegionInfo,
                            (const lldb::SBMemoryRegionInfo &));
  LLDB_REGISTER_METHOD(
      const lldb::SBMemoryRegionInfo &,
      SBMemoryRegionInfo, operator=,(const lldb::SBMemoryRegionInfo &));
  LLDB_REGISTER_METHOD(void, SBMemoryRegionInfo, Clear, ());
  LLDB_REGISTER_METHOD_CONST(
      bool,
      SBMemoryRegionInfo, operator==,(const lldb::SBMemoryRegionInfo &));
  LLDB_REGISTER_METHOD_CONST(
      bool,
      SBMemoryRegionInfo, operator!=,(const lldb::SBMemoryRegionInfo &));
  LLDB_REGISTER_METHOD(lldb::addr_t, SBMemoryRegionInfo, GetRegionBase, ());
  LLDB_REGISTER_METHOD(lldb::addr_t, SBMemoryRegionInfo, GetRegionEnd, ());
  LLDB_REGISTER_METHOD(bool, SBMemoryRegionInfo, IsReadable, ());
  LLDB_REGISTER_METHOD(bool, SBMemoryRegionInfo, IsWritable, ());
  LLDB_REGISTER_METHOD(bool, SBMemoryRegionInfo, IsExecutable, ());
  LLDB_REGISTER_METHOD(bool, SBMemoryRegionInfo, IsMapped, ());
  LLDB_REGISTER_METHOD(const char *, SBMemoryRegionInfo, GetName, ());
  LLDB_REGISTER_METHOD(bool, SBMemoryRegionInfo, GetDescription,
                       (lldb::SBStream &));
}

}
}
