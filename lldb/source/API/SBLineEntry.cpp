//===-- SBLineEntry.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <limits.h>

#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBStream.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Symbol/LineEntry.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

SBLineEntry::SBLineEntry() : m_opaque_up() {}

SBLineEntry::SBLineEntry(const SBLineEntry &rhs) : m_opaque_up() {
  if (rhs.IsValid())
    ref() = rhs.ref();
}

SBLineEntry::SBLineEntry(const lldb_private::LineEntry *lldb_object_ptr)
    : m_opaque_up() {
  if (lldb_object_ptr)
    ref() = *lldb_object_ptr;
}

const SBLineEntry &SBLineEntry::operator=(const SBLineEntry &rhs) {
  if (this != &rhs) {
    if (rhs.IsValid())
      ref() = rhs.ref();
    else
      m_opaque_up.reset();
  }
  return *this;
}

void SBLineEntry::SetLineEntry(const lldb_private::LineEntry &lldb_object_ref) {
  ref() = lldb_object_ref;
}

SBLineEntry::~SBLineEntry() {}

SBAddress SBLineEntry::GetStartAddress() const {
  SBAddress sb_address;
  if (m_opaque_up)
    sb_address.SetAddress(&m_opaque_up->range.GetBaseAddress());

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log) {
    StreamString sstr;
    const Address *addr = sb_address.get();
    if (addr)
      addr->Dump(&sstr, NULL, Address::DumpStyleModuleWithFileAddress,
                 Address::DumpStyleInvalid, 4);
    log->Printf("SBLineEntry(%p)::GetStartAddress () => SBAddress (%p): %s",
                static_cast<void *>(m_opaque_up.get()),
                static_cast<void *>(sb_address.get()), sstr.GetData());
  }

  return sb_address;
}

SBAddress SBLineEntry::GetEndAddress() const {
  SBAddress sb_address;
  if (m_opaque_up) {
    sb_address.SetAddress(&m_opaque_up->range.GetBaseAddress());
    sb_address.OffsetAddress(m_opaque_up->range.GetByteSize());
  }
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log) {
    StreamString sstr;
    const Address *addr = sb_address.get();
    if (addr)
      addr->Dump(&sstr, NULL, Address::DumpStyleModuleWithFileAddress,
                 Address::DumpStyleInvalid, 4);
    log->Printf("SBLineEntry(%p)::GetEndAddress () => SBAddress (%p): %s",
                static_cast<void *>(m_opaque_up.get()),
                static_cast<void *>(sb_address.get()), sstr.GetData());
  }
  return sb_address;
}

bool SBLineEntry::IsValid() const {
  return m_opaque_up.get() && m_opaque_up->IsValid();
}

SBFileSpec SBLineEntry::GetFileSpec() const {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  SBFileSpec sb_file_spec;
  if (m_opaque_up.get() && m_opaque_up->file)
    sb_file_spec.SetFileSpec(m_opaque_up->file);

  if (log) {
    SBStream sstr;
    sb_file_spec.GetDescription(sstr);
    log->Printf("SBLineEntry(%p)::GetFileSpec () => SBFileSpec(%p): %s",
                static_cast<void *>(m_opaque_up.get()),
                static_cast<const void *>(sb_file_spec.get()), sstr.GetData());
  }

  return sb_file_spec;
}

uint32_t SBLineEntry::GetLine() const {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  uint32_t line = 0;
  if (m_opaque_up)
    line = m_opaque_up->line;

  if (log)
    log->Printf("SBLineEntry(%p)::GetLine () => %u",
                static_cast<void *>(m_opaque_up.get()), line);

  return line;
}

uint32_t SBLineEntry::GetColumn() const {
  if (m_opaque_up)
    return m_opaque_up->column;
  return 0;
}

void SBLineEntry::SetFileSpec(lldb::SBFileSpec filespec) {
  if (filespec.IsValid())
    ref().file = filespec.ref();
  else
    ref().file.Clear();
}
void SBLineEntry::SetLine(uint32_t line) { ref().line = line; }

void SBLineEntry::SetColumn(uint32_t column) { ref().line = column; }

bool SBLineEntry::operator==(const SBLineEntry &rhs) const {
  lldb_private::LineEntry *lhs_ptr = m_opaque_up.get();
  lldb_private::LineEntry *rhs_ptr = rhs.m_opaque_up.get();

  if (lhs_ptr && rhs_ptr)
    return lldb_private::LineEntry::Compare(*lhs_ptr, *rhs_ptr) == 0;

  return lhs_ptr == rhs_ptr;
}

bool SBLineEntry::operator!=(const SBLineEntry &rhs) const {
  lldb_private::LineEntry *lhs_ptr = m_opaque_up.get();
  lldb_private::LineEntry *rhs_ptr = rhs.m_opaque_up.get();

  if (lhs_ptr && rhs_ptr)
    return lldb_private::LineEntry::Compare(*lhs_ptr, *rhs_ptr) != 0;

  return lhs_ptr != rhs_ptr;
}

const lldb_private::LineEntry *SBLineEntry::operator->() const {
  return m_opaque_up.get();
}

lldb_private::LineEntry &SBLineEntry::ref() {
  if (m_opaque_up == NULL)
    m_opaque_up.reset(new lldb_private::LineEntry());
  return *m_opaque_up;
}

const lldb_private::LineEntry &SBLineEntry::ref() const { return *m_opaque_up; }

bool SBLineEntry::GetDescription(SBStream &description) {
  Stream &strm = description.ref();

  if (m_opaque_up) {
    char file_path[PATH_MAX * 2];
    m_opaque_up->file.GetPath(file_path, sizeof(file_path));
    strm.Printf("%s:%u", file_path, GetLine());
    if (GetColumn() > 0)
      strm.Printf(":%u", GetColumn());
  } else
    strm.PutCString("No value");

  return true;
}

lldb_private::LineEntry *SBLineEntry::get() { return m_opaque_up.get(); }
