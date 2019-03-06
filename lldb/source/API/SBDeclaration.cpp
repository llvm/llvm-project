//===-- SBDeclaration.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBDeclaration.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBStream.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Symbol/Declaration.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"

#include <limits.h>

using namespace lldb;
using namespace lldb_private;

SBDeclaration::SBDeclaration() : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBDeclaration);
}

SBDeclaration::SBDeclaration(const SBDeclaration &rhs) : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR(SBDeclaration, (const lldb::SBDeclaration &), rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

SBDeclaration::SBDeclaration(const lldb_private::Declaration *lldb_object_ptr)
    : m_opaque_up() {
  if (lldb_object_ptr)
    m_opaque_up = llvm::make_unique<Declaration>(*lldb_object_ptr);
}

const SBDeclaration &SBDeclaration::operator=(const SBDeclaration &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBDeclaration &,
                     SBDeclaration, operator=,(const lldb::SBDeclaration &),
                     rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

void SBDeclaration::SetDeclaration(
    const lldb_private::Declaration &lldb_object_ref) {
  ref() = lldb_object_ref;
}

SBDeclaration::~SBDeclaration() {}

bool SBDeclaration::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBDeclaration, IsValid);

  return m_opaque_up.get() && m_opaque_up->IsValid();
}

SBFileSpec SBDeclaration::GetFileSpec() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBFileSpec, SBDeclaration,
                                   GetFileSpec);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  SBFileSpec sb_file_spec;
  if (m_opaque_up.get() && m_opaque_up->GetFile())
    sb_file_spec.SetFileSpec(m_opaque_up->GetFile());

  if (log) {
    SBStream sstr;
    sb_file_spec.GetDescription(sstr);
    log->Printf("SBLineEntry(%p)::GetFileSpec () => SBFileSpec(%p): %s",
                static_cast<void *>(m_opaque_up.get()),
                static_cast<const void *>(sb_file_spec.get()), sstr.GetData());
  }

  return LLDB_RECORD_RESULT(sb_file_spec);
}

uint32_t SBDeclaration::GetLine() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBDeclaration, GetLine);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  uint32_t line = 0;
  if (m_opaque_up)
    line = m_opaque_up->GetLine();

  if (log)
    log->Printf("SBLineEntry(%p)::GetLine () => %u",
                static_cast<void *>(m_opaque_up.get()), line);

  return line;
}

uint32_t SBDeclaration::GetColumn() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBDeclaration, GetColumn);

  if (m_opaque_up)
    return m_opaque_up->GetColumn();
  return 0;
}

void SBDeclaration::SetFileSpec(lldb::SBFileSpec filespec) {
  LLDB_RECORD_METHOD(void, SBDeclaration, SetFileSpec, (lldb::SBFileSpec),
                     filespec);

  if (filespec.IsValid())
    ref().SetFile(filespec.ref());
  else
    ref().SetFile(FileSpec());
}
void SBDeclaration::SetLine(uint32_t line) {
  LLDB_RECORD_METHOD(void, SBDeclaration, SetLine, (uint32_t), line);

  ref().SetLine(line);
}

void SBDeclaration::SetColumn(uint32_t column) {
  LLDB_RECORD_METHOD(void, SBDeclaration, SetColumn, (uint32_t), column);

  ref().SetColumn(column);
}

bool SBDeclaration::operator==(const SBDeclaration &rhs) const {
  LLDB_RECORD_METHOD_CONST(
      bool, SBDeclaration, operator==,(const lldb::SBDeclaration &), rhs);

  lldb_private::Declaration *lhs_ptr = m_opaque_up.get();
  lldb_private::Declaration *rhs_ptr = rhs.m_opaque_up.get();

  if (lhs_ptr && rhs_ptr)
    return lldb_private::Declaration::Compare(*lhs_ptr, *rhs_ptr) == 0;

  return lhs_ptr == rhs_ptr;
}

bool SBDeclaration::operator!=(const SBDeclaration &rhs) const {
  LLDB_RECORD_METHOD_CONST(
      bool, SBDeclaration, operator!=,(const lldb::SBDeclaration &), rhs);

  lldb_private::Declaration *lhs_ptr = m_opaque_up.get();
  lldb_private::Declaration *rhs_ptr = rhs.m_opaque_up.get();

  if (lhs_ptr && rhs_ptr)
    return lldb_private::Declaration::Compare(*lhs_ptr, *rhs_ptr) != 0;

  return lhs_ptr != rhs_ptr;
}

const lldb_private::Declaration *SBDeclaration::operator->() const {
  return m_opaque_up.get();
}

lldb_private::Declaration &SBDeclaration::ref() {
  if (m_opaque_up == NULL)
    m_opaque_up.reset(new lldb_private::Declaration());
  return *m_opaque_up;
}

const lldb_private::Declaration &SBDeclaration::ref() const {
  return *m_opaque_up;
}

bool SBDeclaration::GetDescription(SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBDeclaration, GetDescription, (lldb::SBStream &),
                     description);

  Stream &strm = description.ref();

  if (m_opaque_up) {
    char file_path[PATH_MAX * 2];
    m_opaque_up->GetFile().GetPath(file_path, sizeof(file_path));
    strm.Printf("%s:%u", file_path, GetLine());
    if (GetColumn() > 0)
      strm.Printf(":%u", GetColumn());
  } else
    strm.PutCString("No value");

  return true;
}

lldb_private::Declaration *SBDeclaration::get() { return m_opaque_up.get(); }
