//===-- SBFileSpecList.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBFileSpecList.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Stream.h"

#include <limits.h>

using namespace lldb;
using namespace lldb_private;

SBFileSpecList::SBFileSpecList() : m_opaque_up(new FileSpecList()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBFileSpecList);
}

SBFileSpecList::SBFileSpecList(const SBFileSpecList &rhs) : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR(SBFileSpecList, (const lldb::SBFileSpecList &), rhs);


  m_opaque_up = clone(rhs.m_opaque_up);
}

SBFileSpecList::~SBFileSpecList() = default;

const SBFileSpecList &SBFileSpecList::operator=(const SBFileSpecList &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBFileSpecList &,
                     SBFileSpecList, operator=,(const lldb::SBFileSpecList &),
                     rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return LLDB_RECORD_RESULT(*this);
}

uint32_t SBFileSpecList::GetSize() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBFileSpecList, GetSize);

  return m_opaque_up->GetSize();
}

void SBFileSpecList::Append(const SBFileSpec &sb_file) {
  LLDB_RECORD_METHOD(void, SBFileSpecList, Append, (const lldb::SBFileSpec &),
                     sb_file);

  m_opaque_up->Append(sb_file.ref());
}

bool SBFileSpecList::AppendIfUnique(const SBFileSpec &sb_file) {
  LLDB_RECORD_METHOD(bool, SBFileSpecList, AppendIfUnique,
                     (const lldb::SBFileSpec &), sb_file);

  return m_opaque_up->AppendIfUnique(sb_file.ref());
}

void SBFileSpecList::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBFileSpecList, Clear);

  m_opaque_up->Clear();
}

uint32_t SBFileSpecList::FindFileIndex(uint32_t idx, const SBFileSpec &sb_file,
                                       bool full) {
  LLDB_RECORD_METHOD(uint32_t, SBFileSpecList, FindFileIndex,
                     (uint32_t, const lldb::SBFileSpec &, bool), idx, sb_file,
                     full);

  return m_opaque_up->FindFileIndex(idx, sb_file.ref(), full);
}

const SBFileSpec SBFileSpecList::GetFileSpecAtIndex(uint32_t idx) const {
  LLDB_RECORD_METHOD_CONST(const lldb::SBFileSpec, SBFileSpecList,
                           GetFileSpecAtIndex, (uint32_t), idx);

  SBFileSpec new_spec;
  new_spec.SetFileSpec(m_opaque_up->GetFileSpecAtIndex(idx));
  return LLDB_RECORD_RESULT(new_spec);
}

const lldb_private::FileSpecList *SBFileSpecList::operator->() const {
  return m_opaque_up.get();
}

const lldb_private::FileSpecList *SBFileSpecList::get() const {
  return m_opaque_up.get();
}

const lldb_private::FileSpecList &SBFileSpecList::operator*() const {
  return *m_opaque_up;
}

const lldb_private::FileSpecList &SBFileSpecList::ref() const {
  return *m_opaque_up;
}

bool SBFileSpecList::GetDescription(SBStream &description) const {
  LLDB_RECORD_METHOD_CONST(bool, SBFileSpecList, GetDescription,
                           (lldb::SBStream &), description);

  Stream &strm = description.ref();

  if (m_opaque_up) {
    uint32_t num_files = m_opaque_up->GetSize();
    strm.Printf("%d files: ", num_files);
    for (uint32_t i = 0; i < num_files; i++) {
      char path[PATH_MAX];
      if (m_opaque_up->GetFileSpecAtIndex(i).GetPath(path, sizeof(path)))
        strm.Printf("\n    %s", path);
    }
  } else
    strm.PutCString("No value");

  return true;
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBFileSpecList>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBFileSpecList, ());
  LLDB_REGISTER_CONSTRUCTOR(SBFileSpecList, (const lldb::SBFileSpecList &));
  LLDB_REGISTER_METHOD(
      const lldb::SBFileSpecList &,
      SBFileSpecList, operator=,(const lldb::SBFileSpecList &));
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBFileSpecList, GetSize, ());
  LLDB_REGISTER_METHOD(void, SBFileSpecList, Append,
                       (const lldb::SBFileSpec &));
  LLDB_REGISTER_METHOD(bool, SBFileSpecList, AppendIfUnique,
                       (const lldb::SBFileSpec &));
  LLDB_REGISTER_METHOD(void, SBFileSpecList, Clear, ());
  LLDB_REGISTER_METHOD(uint32_t, SBFileSpecList, FindFileIndex,
                       (uint32_t, const lldb::SBFileSpec &, bool));
  LLDB_REGISTER_METHOD_CONST(const lldb::SBFileSpec, SBFileSpecList,
                             GetFileSpecAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD_CONST(bool, SBFileSpecList, GetDescription,
                             (lldb::SBStream &));
}

}
}
