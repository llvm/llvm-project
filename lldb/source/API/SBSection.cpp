//===-- SBSection.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBSection.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBTarget.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

SBSection::SBSection() : m_opaque_wp() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBSection);
}

SBSection::SBSection(const SBSection &rhs) : m_opaque_wp(rhs.m_opaque_wp) {
  LLDB_RECORD_CONSTRUCTOR(SBSection, (const lldb::SBSection &), rhs);
}

SBSection::SBSection(const lldb::SectionSP &section_sp)
    : m_opaque_wp() // Don't init with section_sp otherwise this will throw if
                    // section_sp doesn't contain a valid Section *
{
  if (section_sp)
    m_opaque_wp = section_sp;
}

const SBSection &SBSection::operator=(const SBSection &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBSection &,
                     SBSection, operator=,(const lldb::SBSection &), rhs);

  m_opaque_wp = rhs.m_opaque_wp;
  return LLDB_RECORD_RESULT(*this);
}

SBSection::~SBSection() {}

bool SBSection::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBSection, IsValid);
  return this->operator bool();
}
SBSection::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBSection, operator bool);

  SectionSP section_sp(GetSP());
  return section_sp && section_sp->GetModule().get() != nullptr;
}

const char *SBSection::GetName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBSection, GetName);

  SectionSP section_sp(GetSP());
  if (section_sp)
    return section_sp->GetName().GetCString();
  return nullptr;
}

lldb::SBSection SBSection::GetParent() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBSection, SBSection, GetParent);

  lldb::SBSection sb_section;
  SectionSP section_sp(GetSP());
  if (section_sp) {
    SectionSP parent_section_sp(section_sp->GetParent());
    if (parent_section_sp)
      sb_section.SetSP(parent_section_sp);
  }
  return LLDB_RECORD_RESULT(sb_section);
}

lldb::SBSection SBSection::FindSubSection(const char *sect_name) {
  LLDB_RECORD_METHOD(lldb::SBSection, SBSection, FindSubSection, (const char *),
                     sect_name);

  lldb::SBSection sb_section;
  if (sect_name) {
    SectionSP section_sp(GetSP());
    if (section_sp) {
      ConstString const_sect_name(sect_name);
      sb_section.SetSP(
          section_sp->GetChildren().FindSectionByName(const_sect_name));
    }
  }
  return LLDB_RECORD_RESULT(sb_section);
}

size_t SBSection::GetNumSubSections() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBSection, GetNumSubSections);

  SectionSP section_sp(GetSP());
  if (section_sp)
    return section_sp->GetChildren().GetSize();
  return 0;
}

lldb::SBSection SBSection::GetSubSectionAtIndex(size_t idx) {
  LLDB_RECORD_METHOD(lldb::SBSection, SBSection, GetSubSectionAtIndex, (size_t),
                     idx);

  lldb::SBSection sb_section;
  SectionSP section_sp(GetSP());
  if (section_sp)
    sb_section.SetSP(section_sp->GetChildren().GetSectionAtIndex(idx));
  return LLDB_RECORD_RESULT(sb_section);
}

lldb::SectionSP SBSection::GetSP() const { return m_opaque_wp.lock(); }

void SBSection::SetSP(const lldb::SectionSP &section_sp) {
  m_opaque_wp = section_sp;
}

lldb::addr_t SBSection::GetFileAddress() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::addr_t, SBSection, GetFileAddress);

  lldb::addr_t file_addr = LLDB_INVALID_ADDRESS;
  SectionSP section_sp(GetSP());
  if (section_sp)
    return section_sp->GetFileAddress();
  return file_addr;
}

lldb::addr_t SBSection::GetLoadAddress(lldb::SBTarget &sb_target) {
  LLDB_RECORD_METHOD(lldb::addr_t, SBSection, GetLoadAddress,
                     (lldb::SBTarget &), sb_target);

  TargetSP target_sp(sb_target.GetSP());
  if (target_sp) {
    SectionSP section_sp(GetSP());
    if (section_sp)
      return section_sp->GetLoadBaseAddress(target_sp.get());
  }
  return LLDB_INVALID_ADDRESS;
}

lldb::addr_t SBSection::GetByteSize() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::addr_t, SBSection, GetByteSize);

  SectionSP section_sp(GetSP());
  if (section_sp)
    return section_sp->GetByteSize();
  return 0;
}

uint64_t SBSection::GetFileOffset() {
  LLDB_RECORD_METHOD_NO_ARGS(uint64_t, SBSection, GetFileOffset);

  SectionSP section_sp(GetSP());
  if (section_sp) {
    ModuleSP module_sp(section_sp->GetModule());
    if (module_sp) {
      ObjectFile *objfile = module_sp->GetObjectFile();
      if (objfile)
        return objfile->GetFileOffset() + section_sp->GetFileOffset();
    }
  }
  return UINT64_MAX;
}

uint64_t SBSection::GetFileByteSize() {
  LLDB_RECORD_METHOD_NO_ARGS(uint64_t, SBSection, GetFileByteSize);

  SectionSP section_sp(GetSP());
  if (section_sp)
    return section_sp->GetFileSize();
  return 0;
}

SBData SBSection::GetSectionData() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBData, SBSection, GetSectionData);

  return LLDB_RECORD_RESULT(GetSectionData(0, UINT64_MAX));
}

SBData SBSection::GetSectionData(uint64_t offset, uint64_t size) {
  LLDB_RECORD_METHOD(lldb::SBData, SBSection, GetSectionData,
                     (uint64_t, uint64_t), offset, size);

  SBData sb_data;
  SectionSP section_sp(GetSP());
  if (section_sp) {
    const uint64_t sect_file_size = section_sp->GetFileSize();
    if (sect_file_size > 0) {
      ModuleSP module_sp(section_sp->GetModule());
      if (module_sp) {
        ObjectFile *objfile = module_sp->GetObjectFile();
        if (objfile) {
          const uint64_t sect_file_offset =
              objfile->GetFileOffset() + section_sp->GetFileOffset();
          const uint64_t file_offset = sect_file_offset + offset;
          uint64_t file_size = size;
          if (file_size == UINT64_MAX) {
            file_size = section_sp->GetByteSize();
            if (file_size > offset)
              file_size -= offset;
            else
              file_size = 0;
          }
          auto data_buffer_sp = FileSystem::Instance().CreateDataBuffer(
              objfile->GetFileSpec().GetPath(), file_size, file_offset);
          if (data_buffer_sp && data_buffer_sp->GetByteSize() > 0) {
            DataExtractorSP data_extractor_sp(
                new DataExtractor(data_buffer_sp, objfile->GetByteOrder(),
                                  objfile->GetAddressByteSize()));

            sb_data.SetOpaque(data_extractor_sp);
          }
        }
      }
    }
  }
  return LLDB_RECORD_RESULT(sb_data);
}

SectionType SBSection::GetSectionType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SectionType, SBSection, GetSectionType);

  SectionSP section_sp(GetSP());
  if (section_sp.get())
    return section_sp->GetType();
  return eSectionTypeInvalid;
}

uint32_t SBSection::GetPermissions() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBSection, GetPermissions);

  SectionSP section_sp(GetSP());
  if (section_sp)
    return section_sp->GetPermissions();
  return 0;
}

uint32_t SBSection::GetTargetByteSize() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBSection, GetTargetByteSize);

  SectionSP section_sp(GetSP());
  if (section_sp.get())
    return section_sp->GetTargetByteSize();
  return 0;
}

bool SBSection::operator==(const SBSection &rhs) {
  LLDB_RECORD_METHOD(bool, SBSection, operator==,(const lldb::SBSection &),
                     rhs);

  SectionSP lhs_section_sp(GetSP());
  SectionSP rhs_section_sp(rhs.GetSP());
  if (lhs_section_sp && rhs_section_sp)
    return lhs_section_sp == rhs_section_sp;
  return false;
}

bool SBSection::operator!=(const SBSection &rhs) {
  LLDB_RECORD_METHOD(bool, SBSection, operator!=,(const lldb::SBSection &),
                     rhs);

  SectionSP lhs_section_sp(GetSP());
  SectionSP rhs_section_sp(rhs.GetSP());
  return lhs_section_sp != rhs_section_sp;
}

bool SBSection::GetDescription(SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBSection, GetDescription, (lldb::SBStream &),
                     description);

  Stream &strm = description.ref();

  SectionSP section_sp(GetSP());
  if (section_sp) {
    const addr_t file_addr = section_sp->GetFileAddress();
    strm.Printf("[0x%16.16" PRIx64 "-0x%16.16" PRIx64 ") ", file_addr,
                file_addr + section_sp->GetByteSize());
    section_sp->DumpName(&strm);
  } else {
    strm.PutCString("No value");
  }

  return true;
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBSection>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBSection, ());
  LLDB_REGISTER_CONSTRUCTOR(SBSection, (const lldb::SBSection &));
  LLDB_REGISTER_METHOD(const lldb::SBSection &,
                       SBSection, operator=,(const lldb::SBSection &));
  LLDB_REGISTER_METHOD_CONST(bool, SBSection, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBSection, operator bool, ());
  LLDB_REGISTER_METHOD(const char *, SBSection, GetName, ());
  LLDB_REGISTER_METHOD(lldb::SBSection, SBSection, GetParent, ());
  LLDB_REGISTER_METHOD(lldb::SBSection, SBSection, FindSubSection,
                       (const char *));
  LLDB_REGISTER_METHOD(size_t, SBSection, GetNumSubSections, ());
  LLDB_REGISTER_METHOD(lldb::SBSection, SBSection, GetSubSectionAtIndex,
                       (size_t));
  LLDB_REGISTER_METHOD(lldb::addr_t, SBSection, GetFileAddress, ());
  LLDB_REGISTER_METHOD(lldb::addr_t, SBSection, GetLoadAddress,
                       (lldb::SBTarget &));
  LLDB_REGISTER_METHOD(lldb::addr_t, SBSection, GetByteSize, ());
  LLDB_REGISTER_METHOD(uint64_t, SBSection, GetFileOffset, ());
  LLDB_REGISTER_METHOD(uint64_t, SBSection, GetFileByteSize, ());
  LLDB_REGISTER_METHOD(lldb::SBData, SBSection, GetSectionData, ());
  LLDB_REGISTER_METHOD(lldb::SBData, SBSection, GetSectionData,
                       (uint64_t, uint64_t));
  LLDB_REGISTER_METHOD(lldb::SectionType, SBSection, GetSectionType, ());
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBSection, GetPermissions, ());
  LLDB_REGISTER_METHOD(uint32_t, SBSection, GetTargetByteSize, ());
  LLDB_REGISTER_METHOD(bool, SBSection, operator==,(const lldb::SBSection &));
  LLDB_REGISTER_METHOD(bool, SBSection, operator!=,(const lldb::SBSection &));
  LLDB_REGISTER_METHOD(bool, SBSection, GetDescription, (lldb::SBStream &));
}

}
}
