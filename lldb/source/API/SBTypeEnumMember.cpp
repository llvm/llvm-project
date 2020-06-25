//===-- SBTypeEnumMember.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBTypeEnumMember.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBType.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Utility/Stream.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

SBTypeEnumMember::SBTypeEnumMember() : m_opaque_sp() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBTypeEnumMember);
}

SBTypeEnumMember::~SBTypeEnumMember() = default;

SBTypeEnumMember::SBTypeEnumMember(
    const lldb::TypeEnumMemberImplSP &enum_member_sp)
    : m_opaque_sp(enum_member_sp) {}

SBTypeEnumMember::SBTypeEnumMember(const SBTypeEnumMember &rhs)
    : m_opaque_sp() {
  LLDB_RECORD_CONSTRUCTOR(SBTypeEnumMember, (const lldb::SBTypeEnumMember &),
                          rhs);

  m_opaque_sp = clone(rhs.m_opaque_sp);
}

SBTypeEnumMember &SBTypeEnumMember::operator=(const SBTypeEnumMember &rhs) {
  LLDB_RECORD_METHOD(
      SBTypeEnumMember &,
      SBTypeEnumMember, operator=,(const lldb::SBTypeEnumMember &), rhs);

  if (this != &rhs)
    m_opaque_sp = clone(rhs.m_opaque_sp);
  return LLDB_RECORD_RESULT(*this);
}

bool SBTypeEnumMember::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeEnumMember, IsValid);
  return this->operator bool();
}
SBTypeEnumMember::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeEnumMember, operator bool);

  return m_opaque_sp.get();
}

const char *SBTypeEnumMember::GetName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBTypeEnumMember, GetName);

  if (m_opaque_sp.get())
    return m_opaque_sp->GetName().GetCString();
  return nullptr;
}

int64_t SBTypeEnumMember::GetValueAsSigned() {
  LLDB_RECORD_METHOD_NO_ARGS(int64_t, SBTypeEnumMember, GetValueAsSigned);

  if (m_opaque_sp.get())
    return m_opaque_sp->GetValueAsSigned();
  return 0;
}

uint64_t SBTypeEnumMember::GetValueAsUnsigned() {
  LLDB_RECORD_METHOD_NO_ARGS(uint64_t, SBTypeEnumMember, GetValueAsUnsigned);

  if (m_opaque_sp.get())
    return m_opaque_sp->GetValueAsUnsigned();
  return 0;
}

SBType SBTypeEnumMember::GetType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBTypeEnumMember, GetType);

  SBType sb_type;
  if (m_opaque_sp.get()) {
    sb_type.SetSP(m_opaque_sp->GetIntegerType());
  }
  return LLDB_RECORD_RESULT(sb_type);
}

void SBTypeEnumMember::reset(TypeEnumMemberImpl *type_member_impl) {
  m_opaque_sp.reset(type_member_impl);
}

TypeEnumMemberImpl &SBTypeEnumMember::ref() {
  if (m_opaque_sp.get() == nullptr)
    m_opaque_sp = std::make_shared<TypeEnumMemberImpl>();
  return *m_opaque_sp.get();
}

const TypeEnumMemberImpl &SBTypeEnumMember::ref() const {
  return *m_opaque_sp.get();
}

SBTypeEnumMemberList::SBTypeEnumMemberList()
    : m_opaque_up(new TypeEnumMemberListImpl()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBTypeEnumMemberList);
}

SBTypeEnumMemberList::SBTypeEnumMemberList(const SBTypeEnumMemberList &rhs)
    : m_opaque_up(new TypeEnumMemberListImpl()) {
  LLDB_RECORD_CONSTRUCTOR(SBTypeEnumMemberList,
                          (const lldb::SBTypeEnumMemberList &), rhs);

  for (uint32_t i = 0,
                rhs_size = const_cast<SBTypeEnumMemberList &>(rhs).GetSize();
       i < rhs_size; i++)
    Append(const_cast<SBTypeEnumMemberList &>(rhs).GetTypeEnumMemberAtIndex(i));
}

bool SBTypeEnumMemberList::IsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBTypeEnumMemberList, IsValid);
  return this->operator bool();
}
SBTypeEnumMemberList::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeEnumMemberList, operator bool);

  return (m_opaque_up != nullptr);
}

SBTypeEnumMemberList &SBTypeEnumMemberList::
operator=(const SBTypeEnumMemberList &rhs) {
  LLDB_RECORD_METHOD(
      lldb::SBTypeEnumMemberList &,
      SBTypeEnumMemberList, operator=,(const lldb::SBTypeEnumMemberList &),
      rhs);

  if (this != &rhs) {
    m_opaque_up = std::make_unique<TypeEnumMemberListImpl>();
    for (uint32_t i = 0,
                  rhs_size = const_cast<SBTypeEnumMemberList &>(rhs).GetSize();
         i < rhs_size; i++)
      Append(
          const_cast<SBTypeEnumMemberList &>(rhs).GetTypeEnumMemberAtIndex(i));
  }
  return LLDB_RECORD_RESULT(*this);
}

void SBTypeEnumMemberList::Append(SBTypeEnumMember enum_member) {
  LLDB_RECORD_METHOD(void, SBTypeEnumMemberList, Append,
                     (lldb::SBTypeEnumMember), enum_member);

  if (enum_member.IsValid())
    m_opaque_up->Append(enum_member.m_opaque_sp);
}

SBTypeEnumMember
SBTypeEnumMemberList::GetTypeEnumMemberAtIndex(uint32_t index) {
  LLDB_RECORD_METHOD(lldb::SBTypeEnumMember, SBTypeEnumMemberList,
                     GetTypeEnumMemberAtIndex, (uint32_t), index);

  if (m_opaque_up)
    return LLDB_RECORD_RESULT(
        SBTypeEnumMember(m_opaque_up->GetTypeEnumMemberAtIndex(index)));
  return LLDB_RECORD_RESULT(SBTypeEnumMember());
}

uint32_t SBTypeEnumMemberList::GetSize() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBTypeEnumMemberList, GetSize);

  return m_opaque_up->GetSize();
}

SBTypeEnumMemberList::~SBTypeEnumMemberList() = default;

bool SBTypeEnumMember::GetDescription(
    lldb::SBStream &description, lldb::DescriptionLevel description_level) {
  LLDB_RECORD_METHOD(bool, SBTypeEnumMember, GetDescription,
                     (lldb::SBStream &, lldb::DescriptionLevel), description,
                     description_level);

  Stream &strm = description.ref();

  if (m_opaque_sp.get()) {
    if (m_opaque_sp->GetIntegerType()->GetDescription(strm,
                                                      description_level)) {
      strm.Printf(" %s", m_opaque_sp->GetName().GetCString());
    }
  } else {
    strm.PutCString("No value");
  }
  return true;
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBTypeEnumMember>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBTypeEnumMember, ());
  LLDB_REGISTER_CONSTRUCTOR(SBTypeEnumMember,
                            (const lldb::SBTypeEnumMember &));
  LLDB_REGISTER_METHOD(
      lldb::SBTypeEnumMember &,
      SBTypeEnumMember, operator=,(const lldb::SBTypeEnumMember &));
  LLDB_REGISTER_METHOD_CONST(bool, SBTypeEnumMember, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBTypeEnumMember, operator bool, ());
  LLDB_REGISTER_METHOD(const char *, SBTypeEnumMember, GetName, ());
  LLDB_REGISTER_METHOD(int64_t, SBTypeEnumMember, GetValueAsSigned, ());
  LLDB_REGISTER_METHOD(uint64_t, SBTypeEnumMember, GetValueAsUnsigned, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBTypeEnumMember, GetType, ());
  LLDB_REGISTER_CONSTRUCTOR(SBTypeEnumMemberList, ());
  LLDB_REGISTER_CONSTRUCTOR(SBTypeEnumMemberList,
                            (const lldb::SBTypeEnumMemberList &));
  LLDB_REGISTER_METHOD(bool, SBTypeEnumMemberList, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBTypeEnumMemberList, operator bool, ());
  LLDB_REGISTER_METHOD(
      lldb::SBTypeEnumMemberList &,
      SBTypeEnumMemberList, operator=,(const lldb::SBTypeEnumMemberList &));
  LLDB_REGISTER_METHOD(void, SBTypeEnumMemberList, Append,
                       (lldb::SBTypeEnumMember));
  LLDB_REGISTER_METHOD(lldb::SBTypeEnumMember, SBTypeEnumMemberList,
                       GetTypeEnumMemberAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD(uint32_t, SBTypeEnumMemberList, GetSize, ());
  LLDB_REGISTER_METHOD(bool, SBTypeEnumMember, GetDescription,
                       (lldb::SBStream &, lldb::DescriptionLevel));
}

}
}
