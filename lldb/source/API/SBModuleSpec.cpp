//===-- SBModuleSpec.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBModuleSpec.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;

SBModuleSpec::SBModuleSpec() : m_opaque_up(new lldb_private::ModuleSpec()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBModuleSpec);
}

SBModuleSpec::SBModuleSpec(const SBModuleSpec &rhs) : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR(SBModuleSpec, (const lldb::SBModuleSpec &), rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

const SBModuleSpec &SBModuleSpec::operator=(const SBModuleSpec &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBModuleSpec &,
                     SBModuleSpec, operator=,(const lldb::SBModuleSpec &), rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return LLDB_RECORD_RESULT(*this);
}

SBModuleSpec::~SBModuleSpec() = default;

bool SBModuleSpec::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBModuleSpec, IsValid);
  return this->operator bool();
}
SBModuleSpec::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBModuleSpec, operator bool);

  return m_opaque_up->operator bool();
}

void SBModuleSpec::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBModuleSpec, Clear);

  m_opaque_up->Clear();
}

SBFileSpec SBModuleSpec::GetFileSpec() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBFileSpec, SBModuleSpec, GetFileSpec);

  SBFileSpec sb_spec(m_opaque_up->GetFileSpec());
  return LLDB_RECORD_RESULT(sb_spec);
}

void SBModuleSpec::SetFileSpec(const lldb::SBFileSpec &sb_spec) {
  LLDB_RECORD_METHOD(void, SBModuleSpec, SetFileSpec,
                     (const lldb::SBFileSpec &), sb_spec);

  m_opaque_up->GetFileSpec() = *sb_spec;
}

lldb::SBFileSpec SBModuleSpec::GetPlatformFileSpec() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBFileSpec, SBModuleSpec,
                             GetPlatformFileSpec);

  return LLDB_RECORD_RESULT(SBFileSpec(m_opaque_up->GetPlatformFileSpec()));
}

void SBModuleSpec::SetPlatformFileSpec(const lldb::SBFileSpec &sb_spec) {
  LLDB_RECORD_METHOD(void, SBModuleSpec, SetPlatformFileSpec,
                     (const lldb::SBFileSpec &), sb_spec);

  m_opaque_up->GetPlatformFileSpec() = *sb_spec;
}

lldb::SBFileSpec SBModuleSpec::GetSymbolFileSpec() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBFileSpec, SBModuleSpec, GetSymbolFileSpec);

  return LLDB_RECORD_RESULT(SBFileSpec(m_opaque_up->GetSymbolFileSpec()));
}

void SBModuleSpec::SetSymbolFileSpec(const lldb::SBFileSpec &sb_spec) {
  LLDB_RECORD_METHOD(void, SBModuleSpec, SetSymbolFileSpec,
                     (const lldb::SBFileSpec &), sb_spec);

  m_opaque_up->GetSymbolFileSpec() = *sb_spec;
}

const char *SBModuleSpec::GetObjectName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBModuleSpec, GetObjectName);

  return m_opaque_up->GetObjectName().GetCString();
}

void SBModuleSpec::SetObjectName(const char *name) {
  LLDB_RECORD_METHOD(void, SBModuleSpec, SetObjectName, (const char *), name);

  m_opaque_up->GetObjectName().SetCString(name);
}

const char *SBModuleSpec::GetTriple() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBModuleSpec, GetTriple);

  std::string triple(m_opaque_up->GetArchitecture().GetTriple().str());
  // Unique the string so we don't run into ownership issues since the const
  // strings put the string into the string pool once and the strings never
  // comes out
  ConstString const_triple(triple.c_str());
  return const_triple.GetCString();
}

void SBModuleSpec::SetTriple(const char *triple) {
  LLDB_RECORD_METHOD(void, SBModuleSpec, SetTriple, (const char *), triple);

  m_opaque_up->GetArchitecture().SetTriple(triple);
}

const uint8_t *SBModuleSpec::GetUUIDBytes() {
  return m_opaque_up->GetUUID().GetBytes().data();
}

size_t SBModuleSpec::GetUUIDLength() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBModuleSpec, GetUUIDLength);

  return m_opaque_up->GetUUID().GetBytes().size();
}

bool SBModuleSpec::SetUUIDBytes(const uint8_t *uuid, size_t uuid_len) {
  m_opaque_up->GetUUID() = UUID::fromOptionalData(uuid, uuid_len);
  return m_opaque_up->GetUUID().IsValid();
}

bool SBModuleSpec::GetDescription(lldb::SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBModuleSpec, GetDescription, (lldb::SBStream &),
                     description);

  m_opaque_up->Dump(description.ref());
  return true;
}

SBModuleSpecList::SBModuleSpecList() : m_opaque_up(new ModuleSpecList()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBModuleSpecList);
}

SBModuleSpecList::SBModuleSpecList(const SBModuleSpecList &rhs)
    : m_opaque_up(new ModuleSpecList(*rhs.m_opaque_up)) {
  LLDB_RECORD_CONSTRUCTOR(SBModuleSpecList, (const lldb::SBModuleSpecList &),
                          rhs);
}

SBModuleSpecList &SBModuleSpecList::operator=(const SBModuleSpecList &rhs) {
  LLDB_RECORD_METHOD(
      lldb::SBModuleSpecList &,
      SBModuleSpecList, operator=,(const lldb::SBModuleSpecList &), rhs);

  if (this != &rhs)
    *m_opaque_up = *rhs.m_opaque_up;
  return LLDB_RECORD_RESULT(*this);
}

SBModuleSpecList::~SBModuleSpecList() = default;

SBModuleSpecList SBModuleSpecList::GetModuleSpecifications(const char *path) {
  LLDB_RECORD_STATIC_METHOD(lldb::SBModuleSpecList, SBModuleSpecList,
                            GetModuleSpecifications, (const char *), path);

  SBModuleSpecList specs;
  FileSpec file_spec(path);
  FileSystem::Instance().Resolve(file_spec);
  Host::ResolveExecutableInBundle(file_spec);
  ObjectFile::GetModuleSpecifications(file_spec, 0, 0, *specs.m_opaque_up);
  return LLDB_RECORD_RESULT(specs);
}

void SBModuleSpecList::Append(const SBModuleSpec &spec) {
  LLDB_RECORD_METHOD(void, SBModuleSpecList, Append,
                     (const lldb::SBModuleSpec &), spec);

  m_opaque_up->Append(*spec.m_opaque_up);
}

void SBModuleSpecList::Append(const SBModuleSpecList &spec_list) {
  LLDB_RECORD_METHOD(void, SBModuleSpecList, Append,
                     (const lldb::SBModuleSpecList &), spec_list);

  m_opaque_up->Append(*spec_list.m_opaque_up);
}

size_t SBModuleSpecList::GetSize() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBModuleSpecList, GetSize);

  return m_opaque_up->GetSize();
}

SBModuleSpec SBModuleSpecList::GetSpecAtIndex(size_t i) {
  LLDB_RECORD_METHOD(lldb::SBModuleSpec, SBModuleSpecList, GetSpecAtIndex,
                     (size_t), i);

  SBModuleSpec sb_module_spec;
  m_opaque_up->GetModuleSpecAtIndex(i, *sb_module_spec.m_opaque_up);
  return LLDB_RECORD_RESULT(sb_module_spec);
}

SBModuleSpec
SBModuleSpecList::FindFirstMatchingSpec(const SBModuleSpec &match_spec) {
  LLDB_RECORD_METHOD(lldb::SBModuleSpec, SBModuleSpecList,
                     FindFirstMatchingSpec, (const lldb::SBModuleSpec &),
                     match_spec);

  SBModuleSpec sb_module_spec;
  m_opaque_up->FindMatchingModuleSpec(*match_spec.m_opaque_up,
                                      *sb_module_spec.m_opaque_up);
  return LLDB_RECORD_RESULT(sb_module_spec);
}

SBModuleSpecList
SBModuleSpecList::FindMatchingSpecs(const SBModuleSpec &match_spec) {
  LLDB_RECORD_METHOD(lldb::SBModuleSpecList, SBModuleSpecList,
                     FindMatchingSpecs, (const lldb::SBModuleSpec &),
                     match_spec);

  SBModuleSpecList specs;
  m_opaque_up->FindMatchingModuleSpecs(*match_spec.m_opaque_up,
                                       *specs.m_opaque_up);
  return LLDB_RECORD_RESULT(specs);
}

bool SBModuleSpecList::GetDescription(lldb::SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBModuleSpecList, GetDescription, (lldb::SBStream &),
                     description);

  m_opaque_up->Dump(description.ref());
  return true;
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBModuleSpec>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBModuleSpec, ());
  LLDB_REGISTER_CONSTRUCTOR(SBModuleSpec, (const lldb::SBModuleSpec &));
  LLDB_REGISTER_METHOD(const lldb::SBModuleSpec &,
                       SBModuleSpec, operator=,(const lldb::SBModuleSpec &));
  LLDB_REGISTER_METHOD_CONST(bool, SBModuleSpec, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBModuleSpec, operator bool, ());
  LLDB_REGISTER_METHOD(void, SBModuleSpec, Clear, ());
  LLDB_REGISTER_METHOD(lldb::SBFileSpec, SBModuleSpec, GetFileSpec, ());
  LLDB_REGISTER_METHOD(void, SBModuleSpec, SetFileSpec,
                       (const lldb::SBFileSpec &));
  LLDB_REGISTER_METHOD(lldb::SBFileSpec, SBModuleSpec, GetPlatformFileSpec,
                       ());
  LLDB_REGISTER_METHOD(void, SBModuleSpec, SetPlatformFileSpec,
                       (const lldb::SBFileSpec &));
  LLDB_REGISTER_METHOD(lldb::SBFileSpec, SBModuleSpec, GetSymbolFileSpec, ());
  LLDB_REGISTER_METHOD(void, SBModuleSpec, SetSymbolFileSpec,
                       (const lldb::SBFileSpec &));
  LLDB_REGISTER_METHOD(const char *, SBModuleSpec, GetObjectName, ());
  LLDB_REGISTER_METHOD(void, SBModuleSpec, SetObjectName, (const char *));
  LLDB_REGISTER_METHOD(const char *, SBModuleSpec, GetTriple, ());
  LLDB_REGISTER_METHOD(void, SBModuleSpec, SetTriple, (const char *));
  LLDB_REGISTER_METHOD(size_t, SBModuleSpec, GetUUIDLength, ());
  LLDB_REGISTER_METHOD(bool, SBModuleSpec, GetDescription,
                       (lldb::SBStream &));
  LLDB_REGISTER_CONSTRUCTOR(SBModuleSpecList, ());
  LLDB_REGISTER_CONSTRUCTOR(SBModuleSpecList,
                            (const lldb::SBModuleSpecList &));
  LLDB_REGISTER_METHOD(
      lldb::SBModuleSpecList &,
      SBModuleSpecList, operator=,(const lldb::SBModuleSpecList &));
  LLDB_REGISTER_STATIC_METHOD(lldb::SBModuleSpecList, SBModuleSpecList,
                              GetModuleSpecifications, (const char *));
  LLDB_REGISTER_METHOD(void, SBModuleSpecList, Append,
                       (const lldb::SBModuleSpec &));
  LLDB_REGISTER_METHOD(void, SBModuleSpecList, Append,
                       (const lldb::SBModuleSpecList &));
  LLDB_REGISTER_METHOD(size_t, SBModuleSpecList, GetSize, ());
  LLDB_REGISTER_METHOD(lldb::SBModuleSpec, SBModuleSpecList, GetSpecAtIndex,
                       (size_t));
  LLDB_REGISTER_METHOD(lldb::SBModuleSpec, SBModuleSpecList,
                       FindFirstMatchingSpec, (const lldb::SBModuleSpec &));
  LLDB_REGISTER_METHOD(lldb::SBModuleSpecList, SBModuleSpecList,
                       FindMatchingSpecs, (const lldb::SBModuleSpec &));
  LLDB_REGISTER_METHOD(bool, SBModuleSpecList, GetDescription,
                       (lldb::SBStream &));
}

}
}
