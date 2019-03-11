//===-- SBLineEntry.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBLineEntry.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBStream.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Symbol/LineEntry.h"
#include "lldb/Utility/StreamString.h"

#include <limits.h>

using namespace lldb;
using namespace lldb_private;

SBLineEntry::SBLineEntry() : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBLineEntry);
}

SBLineEntry::SBLineEntry(const SBLineEntry &rhs) : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR(SBLineEntry, (const lldb::SBLineEntry &), rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

SBLineEntry::SBLineEntry(const lldb_private::LineEntry *lldb_object_ptr)
    : m_opaque_up() {
  if (lldb_object_ptr)
    m_opaque_up = llvm::make_unique<LineEntry>(*lldb_object_ptr);
}

const SBLineEntry &SBLineEntry::operator=(const SBLineEntry &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBLineEntry &,
                     SBLineEntry, operator=,(const lldb::SBLineEntry &), rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

void SBLineEntry::SetLineEntry(const lldb_private::LineEntry &lldb_object_ref) {
  m_opaque_up = llvm::make_unique<LineEntry>(lldb_object_ref);
}

SBLineEntry::~SBLineEntry() {}

SBAddress SBLineEntry::GetStartAddress() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBAddress, SBLineEntry,
                                   GetStartAddress);

  SBAddress sb_address;
  if (m_opaque_up)
    sb_address.SetAddress(&m_opaque_up->range.GetBaseAddress());

  return LLDB_RECORD_RESULT(sb_address);
}

SBAddress SBLineEntry::GetEndAddress() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBAddress, SBLineEntry, GetEndAddress);

  SBAddress sb_address;
  if (m_opaque_up) {
    sb_address.SetAddress(&m_opaque_up->range.GetBaseAddress());
    sb_address.OffsetAddress(m_opaque_up->range.GetByteSize());
  }
  return LLDB_RECORD_RESULT(sb_address);
}

bool SBLineEntry::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBLineEntry, IsValid);
  return this->operator bool();
}
SBLineEntry::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBLineEntry, operator bool);

  return m_opaque_up.get() && m_opaque_up->IsValid();
}

SBFileSpec SBLineEntry::GetFileSpec() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBFileSpec, SBLineEntry, GetFileSpec);

  SBFileSpec sb_file_spec;
  if (m_opaque_up.get() && m_opaque_up->file)
    sb_file_spec.SetFileSpec(m_opaque_up->file);

  return LLDB_RECORD_RESULT(sb_file_spec);
}

uint32_t SBLineEntry::GetLine() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBLineEntry, GetLine);

  uint32_t line = 0;
  if (m_opaque_up)
    line = m_opaque_up->line;

  return line;
}

uint32_t SBLineEntry::GetColumn() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBLineEntry, GetColumn);

  if (m_opaque_up)
    return m_opaque_up->column;
  return 0;
}

void SBLineEntry::SetFileSpec(lldb::SBFileSpec filespec) {
  LLDB_RECORD_METHOD(void, SBLineEntry, SetFileSpec, (lldb::SBFileSpec),
                     filespec);

  if (filespec.IsValid())
    ref().file = filespec.ref();
  else
    ref().file.Clear();
}
void SBLineEntry::SetLine(uint32_t line) {
  LLDB_RECORD_METHOD(void, SBLineEntry, SetLine, (uint32_t), line);

  ref().line = line;
}

void SBLineEntry::SetColumn(uint32_t column) {
  LLDB_RECORD_METHOD(void, SBLineEntry, SetColumn, (uint32_t), column);

  ref().line = column;
}

bool SBLineEntry::operator==(const SBLineEntry &rhs) const {
  LLDB_RECORD_METHOD_CONST(
      bool, SBLineEntry, operator==,(const lldb::SBLineEntry &), rhs);

  lldb_private::LineEntry *lhs_ptr = m_opaque_up.get();
  lldb_private::LineEntry *rhs_ptr = rhs.m_opaque_up.get();

  if (lhs_ptr && rhs_ptr)
    return lldb_private::LineEntry::Compare(*lhs_ptr, *rhs_ptr) == 0;

  return lhs_ptr == rhs_ptr;
}

bool SBLineEntry::operator!=(const SBLineEntry &rhs) const {
  LLDB_RECORD_METHOD_CONST(
      bool, SBLineEntry, operator!=,(const lldb::SBLineEntry &), rhs);

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
  LLDB_RECORD_METHOD(bool, SBLineEntry, GetDescription, (lldb::SBStream &),
                     description);

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
