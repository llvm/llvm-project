//===-- SBCompileUnit.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBCompileUnit.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineEntry.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"

using namespace lldb;
using namespace lldb_private;

SBCompileUnit::SBCompileUnit() : m_opaque_ptr(nullptr) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBCompileUnit);
}

SBCompileUnit::SBCompileUnit(lldb_private::CompileUnit *lldb_object_ptr)
    : m_opaque_ptr(lldb_object_ptr) {}

SBCompileUnit::SBCompileUnit(const SBCompileUnit &rhs)
    : m_opaque_ptr(rhs.m_opaque_ptr) {
  LLDB_RECORD_CONSTRUCTOR(SBCompileUnit, (const lldb::SBCompileUnit &), rhs);
}

const SBCompileUnit &SBCompileUnit::operator=(const SBCompileUnit &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBCompileUnit &,
                     SBCompileUnit, operator=,(const lldb::SBCompileUnit &),
                     rhs);

  m_opaque_ptr = rhs.m_opaque_ptr;
  return LLDB_RECORD_RESULT(*this);
}

SBCompileUnit::~SBCompileUnit() { m_opaque_ptr = nullptr; }

SBFileSpec SBCompileUnit::GetFileSpec() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBFileSpec, SBCompileUnit,
                                   GetFileSpec);

  SBFileSpec file_spec;
  if (m_opaque_ptr)
    file_spec.SetFileSpec(m_opaque_ptr->GetPrimaryFile());
  return LLDB_RECORD_RESULT(file_spec);
}

uint32_t SBCompileUnit::GetNumLineEntries() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBCompileUnit, GetNumLineEntries);

  if (m_opaque_ptr) {
    LineTable *line_table = m_opaque_ptr->GetLineTable();
    if (line_table) {
      return line_table->GetSize();
    }
  }
  return 0;
}

SBLineEntry SBCompileUnit::GetLineEntryAtIndex(uint32_t idx) const {
  LLDB_RECORD_METHOD_CONST(lldb::SBLineEntry, SBCompileUnit,
                           GetLineEntryAtIndex, (uint32_t), idx);

  SBLineEntry sb_line_entry;
  if (m_opaque_ptr) {
    LineTable *line_table = m_opaque_ptr->GetLineTable();
    if (line_table) {
      LineEntry line_entry;
      if (line_table->GetLineEntryAtIndex(idx, line_entry))
        sb_line_entry.SetLineEntry(line_entry);
    }
  }

  return LLDB_RECORD_RESULT(sb_line_entry);
}

uint32_t SBCompileUnit::FindLineEntryIndex(uint32_t start_idx, uint32_t line,
                                           SBFileSpec *inline_file_spec) const {
  LLDB_RECORD_METHOD_CONST(uint32_t, SBCompileUnit, FindLineEntryIndex,
                           (uint32_t, uint32_t, lldb::SBFileSpec *), start_idx,
                           line, inline_file_spec);

  const bool exact = true;
  return FindLineEntryIndex(start_idx, line, inline_file_spec, exact);
}

uint32_t SBCompileUnit::FindLineEntryIndex(uint32_t start_idx, uint32_t line,
                                           SBFileSpec *inline_file_spec,
                                           bool exact) const {
  LLDB_RECORD_METHOD_CONST(uint32_t, SBCompileUnit, FindLineEntryIndex,
                           (uint32_t, uint32_t, lldb::SBFileSpec *, bool),
                           start_idx, line, inline_file_spec, exact);

  uint32_t index = UINT32_MAX;
  if (m_opaque_ptr) {
    FileSpec file_spec;
    if (inline_file_spec && inline_file_spec->IsValid())
      file_spec = inline_file_spec->ref();
    else
      file_spec = m_opaque_ptr->GetPrimaryFile();

    LineEntry line_entry;
    index = m_opaque_ptr->FindLineEntry(
        start_idx, line, inline_file_spec ? inline_file_spec->get() : nullptr,
        exact, &line_entry);
  }

  return index;
}

uint32_t SBCompileUnit::GetNumSupportFiles() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBCompileUnit, GetNumSupportFiles);

  if (m_opaque_ptr)
    return m_opaque_ptr->GetSupportFiles().GetSize();

  return 0;
}

lldb::SBTypeList SBCompileUnit::GetTypes(uint32_t type_mask) {
  LLDB_RECORD_METHOD(lldb::SBTypeList, SBCompileUnit, GetTypes, (uint32_t),
                     type_mask);

  SBTypeList sb_type_list;

  if (!m_opaque_ptr)
    return LLDB_RECORD_RESULT(sb_type_list);

  ModuleSP module_sp(m_opaque_ptr->GetModule());
  if (!module_sp)
    return LLDB_RECORD_RESULT(sb_type_list);

  SymbolFile *symfile = module_sp->GetSymbolFile();
  if (!symfile)
    return LLDB_RECORD_RESULT(sb_type_list);

  TypeClass type_class = static_cast<TypeClass>(type_mask);
  TypeList type_list;
  symfile->GetTypes(m_opaque_ptr, type_class, type_list);
  sb_type_list.m_opaque_up->Append(type_list);
  return LLDB_RECORD_RESULT(sb_type_list);
}

SBFileSpec SBCompileUnit::GetSupportFileAtIndex(uint32_t idx) const {
  LLDB_RECORD_METHOD_CONST(lldb::SBFileSpec, SBCompileUnit,
                           GetSupportFileAtIndex, (uint32_t), idx);

  SBFileSpec sb_file_spec;
  if (m_opaque_ptr) {
    FileSpec spec = m_opaque_ptr->GetSupportFiles().GetFileSpecAtIndex(idx);
    sb_file_spec.SetFileSpec(spec);
  }


  return LLDB_RECORD_RESULT(sb_file_spec);
}

uint32_t SBCompileUnit::FindSupportFileIndex(uint32_t start_idx,
                                             const SBFileSpec &sb_file,
                                             bool full) {
  LLDB_RECORD_METHOD(uint32_t, SBCompileUnit, FindSupportFileIndex,
                     (uint32_t, const lldb::SBFileSpec &, bool), start_idx,
                     sb_file, full);

  if (m_opaque_ptr) {
    const FileSpecList &support_files = m_opaque_ptr->GetSupportFiles();
    return support_files.FindFileIndex(start_idx, sb_file.ref(), full);
  }
  return 0;
}

lldb::LanguageType SBCompileUnit::GetLanguage() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::LanguageType, SBCompileUnit, GetLanguage);

  if (m_opaque_ptr)
    return m_opaque_ptr->GetLanguage();
  return lldb::eLanguageTypeUnknown;
}

bool SBCompileUnit::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCompileUnit, IsValid);
  return this->operator bool();
}
SBCompileUnit::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCompileUnit, operator bool);

  return m_opaque_ptr != nullptr;
}

bool SBCompileUnit::operator==(const SBCompileUnit &rhs) const {
  LLDB_RECORD_METHOD_CONST(
      bool, SBCompileUnit, operator==,(const lldb::SBCompileUnit &), rhs);

  return m_opaque_ptr == rhs.m_opaque_ptr;
}

bool SBCompileUnit::operator!=(const SBCompileUnit &rhs) const {
  LLDB_RECORD_METHOD_CONST(
      bool, SBCompileUnit, operator!=,(const lldb::SBCompileUnit &), rhs);

  return m_opaque_ptr != rhs.m_opaque_ptr;
}

const lldb_private::CompileUnit *SBCompileUnit::operator->() const {
  return m_opaque_ptr;
}

const lldb_private::CompileUnit &SBCompileUnit::operator*() const {
  return *m_opaque_ptr;
}

lldb_private::CompileUnit *SBCompileUnit::get() { return m_opaque_ptr; }

void SBCompileUnit::reset(lldb_private::CompileUnit *lldb_object_ptr) {
  m_opaque_ptr = lldb_object_ptr;
}

bool SBCompileUnit::GetDescription(SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBCompileUnit, GetDescription, (lldb::SBStream &),
                     description);

  Stream &strm = description.ref();

  if (m_opaque_ptr) {
    m_opaque_ptr->Dump(&strm, false);
  } else
    strm.PutCString("No value");

  return true;
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBCompileUnit>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBCompileUnit, ());
  LLDB_REGISTER_CONSTRUCTOR(SBCompileUnit, (const lldb::SBCompileUnit &));
  LLDB_REGISTER_METHOD(
      const lldb::SBCompileUnit &,
      SBCompileUnit, operator=,(const lldb::SBCompileUnit &));
  LLDB_REGISTER_METHOD_CONST(lldb::SBFileSpec, SBCompileUnit, GetFileSpec,
                             ());
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBCompileUnit, GetNumLineEntries, ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBLineEntry, SBCompileUnit,
                             GetLineEntryAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBCompileUnit, FindLineEntryIndex,
                             (uint32_t, uint32_t, lldb::SBFileSpec *));
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBCompileUnit, FindLineEntryIndex,
                             (uint32_t, uint32_t, lldb::SBFileSpec *, bool));
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBCompileUnit, GetNumSupportFiles, ());
  LLDB_REGISTER_METHOD(lldb::SBTypeList, SBCompileUnit, GetTypes, (uint32_t));
  LLDB_REGISTER_METHOD_CONST(lldb::SBFileSpec, SBCompileUnit,
                             GetSupportFileAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD(uint32_t, SBCompileUnit, FindSupportFileIndex,
                       (uint32_t, const lldb::SBFileSpec &, bool));
  LLDB_REGISTER_METHOD(lldb::LanguageType, SBCompileUnit, GetLanguage, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBCompileUnit, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBCompileUnit, operator bool, ());
  LLDB_REGISTER_METHOD_CONST(
      bool, SBCompileUnit, operator==,(const lldb::SBCompileUnit &));
  LLDB_REGISTER_METHOD_CONST(
      bool, SBCompileUnit, operator!=,(const lldb::SBCompileUnit &));
  LLDB_REGISTER_METHOD(bool, SBCompileUnit, GetDescription,
                       (lldb::SBStream &));
}

}
}
