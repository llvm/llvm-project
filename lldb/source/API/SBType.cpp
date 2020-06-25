//===-- SBType.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBType.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBTypeEnumMember.h"
#include "lldb/Core/Mangled.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Stream.h"

#include "llvm/ADT/APSInt.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

SBType::SBType() : m_opaque_sp() { LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBType); }

SBType::SBType(const CompilerType &type)
    : m_opaque_sp(new TypeImpl(
          CompilerType(type.GetTypeSystem(), type.GetOpaqueQualType()))) {}

SBType::SBType(const lldb::TypeSP &type_sp)
    : m_opaque_sp(new TypeImpl(type_sp)) {}

SBType::SBType(const lldb::TypeImplSP &type_impl_sp)
    : m_opaque_sp(type_impl_sp) {}

SBType::SBType(const SBType &rhs) : m_opaque_sp() {
  LLDB_RECORD_CONSTRUCTOR(SBType, (const lldb::SBType &), rhs);

  if (this != &rhs) {
    m_opaque_sp = rhs.m_opaque_sp;
  }
}

// SBType::SBType (TypeImpl* impl) :
//    m_opaque_up(impl)
//{}
//
bool SBType::operator==(SBType &rhs) {
  LLDB_RECORD_METHOD(bool, SBType, operator==,(lldb::SBType &), rhs);

  if (!IsValid())
    return !rhs.IsValid();

  if (!rhs.IsValid())
    return false;

  return *m_opaque_sp.get() == *rhs.m_opaque_sp.get();
}

bool SBType::operator!=(SBType &rhs) {
  LLDB_RECORD_METHOD(bool, SBType, operator!=,(lldb::SBType &), rhs);

  if (!IsValid())
    return rhs.IsValid();

  if (!rhs.IsValid())
    return true;

  return *m_opaque_sp.get() != *rhs.m_opaque_sp.get();
}

lldb::TypeImplSP SBType::GetSP() { return m_opaque_sp; }

void SBType::SetSP(const lldb::TypeImplSP &type_impl_sp) {
  m_opaque_sp = type_impl_sp;
}

SBType &SBType::operator=(const SBType &rhs) {
  LLDB_RECORD_METHOD(lldb::SBType &, SBType, operator=,(const lldb::SBType &),
                     rhs);

  if (this != &rhs) {
    m_opaque_sp = rhs.m_opaque_sp;
  }
  return LLDB_RECORD_RESULT(*this);
}

SBType::~SBType() = default;

TypeImpl &SBType::ref() {
  if (m_opaque_sp.get() == nullptr)
    m_opaque_sp = std::make_shared<TypeImpl>();
  return *m_opaque_sp;
}

const TypeImpl &SBType::ref() const {
  // "const SBAddress &addr" should already have checked "addr.IsValid()" prior
  // to calling this function. In case you didn't we will assert and die to let
  // you know.
  assert(m_opaque_sp.get());
  return *m_opaque_sp;
}

bool SBType::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBType, IsValid);
  return this->operator bool();
}
SBType::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBType, operator bool);

  if (m_opaque_sp.get() == nullptr)
    return false;

  return m_opaque_sp->IsValid();
}

uint64_t SBType::GetByteSize() {
  LLDB_RECORD_METHOD_NO_ARGS(uint64_t, SBType, GetByteSize);

  if (IsValid())
    if (llvm::Optional<uint64_t> size =
            m_opaque_sp->GetCompilerType(false).GetByteSize(nullptr))
      return *size;
  return 0;
}

bool SBType::IsPointerType() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBType, IsPointerType);

  if (!IsValid())
    return false;
  return m_opaque_sp->GetCompilerType(true).IsPointerType();
}

bool SBType::IsArrayType() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBType, IsArrayType);

  if (!IsValid())
    return false;
  return m_opaque_sp->GetCompilerType(true).IsArrayType(nullptr, nullptr,
                                                        nullptr);
}

bool SBType::IsVectorType() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBType, IsVectorType);

  if (!IsValid())
    return false;
  return m_opaque_sp->GetCompilerType(true).IsVectorType(nullptr, nullptr);
}

bool SBType::IsReferenceType() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBType, IsReferenceType);

  if (!IsValid())
    return false;
  return m_opaque_sp->GetCompilerType(true).IsReferenceType();
}

SBType SBType::GetPointerType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBType, GetPointerType);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBType());

  return LLDB_RECORD_RESULT(
      SBType(TypeImplSP(new TypeImpl(m_opaque_sp->GetPointerType()))));
}

SBType SBType::GetPointeeType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBType, GetPointeeType);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBType());
  return LLDB_RECORD_RESULT(
      SBType(TypeImplSP(new TypeImpl(m_opaque_sp->GetPointeeType()))));
}

SBType SBType::GetReferenceType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBType, GetReferenceType);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBType());
  return LLDB_RECORD_RESULT(
      SBType(TypeImplSP(new TypeImpl(m_opaque_sp->GetReferenceType()))));
}

SBType SBType::GetTypedefedType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBType, GetTypedefedType);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBType());
  return LLDB_RECORD_RESULT(
      SBType(TypeImplSP(new TypeImpl(m_opaque_sp->GetTypedefedType()))));
}

SBType SBType::GetDereferencedType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBType, GetDereferencedType);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBType());
  return LLDB_RECORD_RESULT(
      SBType(TypeImplSP(new TypeImpl(m_opaque_sp->GetDereferencedType()))));
}

SBType SBType::GetArrayElementType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBType, GetArrayElementType);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBType());
  CompilerType canonical_type =
      m_opaque_sp->GetCompilerType(true).GetCanonicalType();
  return LLDB_RECORD_RESULT(
      SBType(TypeImplSP(new TypeImpl(canonical_type.GetArrayElementType()))));
}

SBType SBType::GetArrayType(uint64_t size) {
  LLDB_RECORD_METHOD(lldb::SBType, SBType, GetArrayType, (uint64_t), size);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBType());
  return LLDB_RECORD_RESULT(SBType(TypeImplSP(
      new TypeImpl(m_opaque_sp->GetCompilerType(true).GetArrayType(size)))));
}

SBType SBType::GetVectorElementType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBType, GetVectorElementType);

  SBType type_sb;
  if (IsValid()) {
    CompilerType vector_element_type;
    if (m_opaque_sp->GetCompilerType(true).IsVectorType(&vector_element_type,
                                                        nullptr))
      type_sb.SetSP(TypeImplSP(new TypeImpl(vector_element_type)));
  }
  return LLDB_RECORD_RESULT(type_sb);
}

bool SBType::IsFunctionType() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBType, IsFunctionType);

  if (!IsValid())
    return false;
  return m_opaque_sp->GetCompilerType(true).IsFunctionType();
}

bool SBType::IsPolymorphicClass() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBType, IsPolymorphicClass);

  if (!IsValid())
    return false;
  return m_opaque_sp->GetCompilerType(true).IsPolymorphicClass();
}

bool SBType::IsTypedefType() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBType, IsTypedefType);

  if (!IsValid())
    return false;
  return m_opaque_sp->GetCompilerType(true).IsTypedefType();
}

bool SBType::IsAnonymousType() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBType, IsAnonymousType);

  if (!IsValid())
    return false;
  return m_opaque_sp->GetCompilerType(true).IsAnonymousType();
}

lldb::SBType SBType::GetFunctionReturnType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBType, GetFunctionReturnType);

  if (IsValid()) {
    CompilerType return_type(
        m_opaque_sp->GetCompilerType(true).GetFunctionReturnType());
    if (return_type.IsValid())
      return LLDB_RECORD_RESULT(SBType(return_type));
  }
  return LLDB_RECORD_RESULT(lldb::SBType());
}

lldb::SBTypeList SBType::GetFunctionArgumentTypes() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBTypeList, SBType,
                             GetFunctionArgumentTypes);

  SBTypeList sb_type_list;
  if (IsValid()) {
    CompilerType func_type(m_opaque_sp->GetCompilerType(true));
    size_t count = func_type.GetNumberOfFunctionArguments();
    for (size_t i = 0; i < count; i++) {
      sb_type_list.Append(SBType(func_type.GetFunctionArgumentAtIndex(i)));
    }
  }
  return LLDB_RECORD_RESULT(sb_type_list);
}

uint32_t SBType::GetNumberOfMemberFunctions() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBType, GetNumberOfMemberFunctions);

  if (IsValid()) {
    return m_opaque_sp->GetCompilerType(true).GetNumMemberFunctions();
  }
  return 0;
}

lldb::SBTypeMemberFunction SBType::GetMemberFunctionAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(lldb::SBTypeMemberFunction, SBType,
                     GetMemberFunctionAtIndex, (uint32_t), idx);

  SBTypeMemberFunction sb_func_type;
  if (IsValid())
    sb_func_type.reset(new TypeMemberFunctionImpl(
        m_opaque_sp->GetCompilerType(true).GetMemberFunctionAtIndex(idx)));
  return LLDB_RECORD_RESULT(sb_func_type);
}

lldb::SBType SBType::GetUnqualifiedType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBType, GetUnqualifiedType);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBType());
  return LLDB_RECORD_RESULT(
      SBType(TypeImplSP(new TypeImpl(m_opaque_sp->GetUnqualifiedType()))));
}

lldb::SBType SBType::GetCanonicalType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBType, GetCanonicalType);

  if (IsValid())
    return LLDB_RECORD_RESULT(
        SBType(TypeImplSP(new TypeImpl(m_opaque_sp->GetCanonicalType()))));
  return LLDB_RECORD_RESULT(SBType());
}

lldb::BasicType SBType::GetBasicType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::BasicType, SBType, GetBasicType);

  if (IsValid())
    return m_opaque_sp->GetCompilerType(false).GetBasicTypeEnumeration();
  return eBasicTypeInvalid;
}

SBType SBType::GetBasicType(lldb::BasicType basic_type) {
  LLDB_RECORD_METHOD(lldb::SBType, SBType, GetBasicType, (lldb::BasicType),
                     basic_type);

  if (IsValid() && m_opaque_sp->IsValid())
    return LLDB_RECORD_RESULT(SBType(
        m_opaque_sp->GetTypeSystem(false)->GetBasicTypeFromAST(basic_type)));
  return LLDB_RECORD_RESULT(SBType());
}

uint32_t SBType::GetNumberOfDirectBaseClasses() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBType, GetNumberOfDirectBaseClasses);

  if (IsValid())
    return m_opaque_sp->GetCompilerType(true).GetNumDirectBaseClasses();
  return 0;
}

uint32_t SBType::GetNumberOfVirtualBaseClasses() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBType, GetNumberOfVirtualBaseClasses);

  if (IsValid())
    return m_opaque_sp->GetCompilerType(true).GetNumVirtualBaseClasses();
  return 0;
}

uint32_t SBType::GetNumberOfFields() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBType, GetNumberOfFields);

  if (IsValid())
    return m_opaque_sp->GetCompilerType(true).GetNumFields();
  return 0;
}

bool SBType::GetDescription(SBStream &description,
                            lldb::DescriptionLevel description_level) {
  LLDB_RECORD_METHOD(bool, SBType, GetDescription,
                     (lldb::SBStream &, lldb::DescriptionLevel), description,
                     description_level);

  Stream &strm = description.ref();

  if (m_opaque_sp) {
    m_opaque_sp->GetDescription(strm, description_level);
  } else
    strm.PutCString("No value");

  return true;
}

SBTypeMember SBType::GetDirectBaseClassAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(lldb::SBTypeMember, SBType, GetDirectBaseClassAtIndex,
                     (uint32_t), idx);

  SBTypeMember sb_type_member;
  if (IsValid()) {
    uint32_t bit_offset = 0;
    CompilerType base_class_type =
        m_opaque_sp->GetCompilerType(true).GetDirectBaseClassAtIndex(
            idx, &bit_offset);
    if (base_class_type.IsValid())
      sb_type_member.reset(new TypeMemberImpl(
          TypeImplSP(new TypeImpl(base_class_type)), bit_offset));
  }
  return LLDB_RECORD_RESULT(sb_type_member);
}

SBTypeMember SBType::GetVirtualBaseClassAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(lldb::SBTypeMember, SBType, GetVirtualBaseClassAtIndex,
                     (uint32_t), idx);

  SBTypeMember sb_type_member;
  if (IsValid()) {
    uint32_t bit_offset = 0;
    CompilerType base_class_type =
        m_opaque_sp->GetCompilerType(true).GetVirtualBaseClassAtIndex(
            idx, &bit_offset);
    if (base_class_type.IsValid())
      sb_type_member.reset(new TypeMemberImpl(
          TypeImplSP(new TypeImpl(base_class_type)), bit_offset));
  }
  return LLDB_RECORD_RESULT(sb_type_member);
}

SBTypeEnumMemberList SBType::GetEnumMembers() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBTypeEnumMemberList, SBType,
                             GetEnumMembers);

  SBTypeEnumMemberList sb_enum_member_list;
  if (IsValid()) {
    CompilerType this_type(m_opaque_sp->GetCompilerType(true));
    if (this_type.IsValid()) {
      this_type.ForEachEnumerator([&sb_enum_member_list](
                                      const CompilerType &integer_type,
                                      ConstString name,
                                      const llvm::APSInt &value) -> bool {
        SBTypeEnumMember enum_member(
            lldb::TypeEnumMemberImplSP(new TypeEnumMemberImpl(
                lldb::TypeImplSP(new TypeImpl(integer_type)), name, value)));
        sb_enum_member_list.Append(enum_member);
        return true; // Keep iterating
      });
    }
  }
  return LLDB_RECORD_RESULT(sb_enum_member_list);
}

SBTypeMember SBType::GetFieldAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(lldb::SBTypeMember, SBType, GetFieldAtIndex, (uint32_t),
                     idx);

  SBTypeMember sb_type_member;
  if (IsValid()) {
    CompilerType this_type(m_opaque_sp->GetCompilerType(false));
    if (this_type.IsValid()) {
      uint64_t bit_offset = 0;
      uint32_t bitfield_bit_size = 0;
      bool is_bitfield = false;
      std::string name_sstr;
      CompilerType field_type(this_type.GetFieldAtIndex(
          idx, name_sstr, &bit_offset, &bitfield_bit_size, &is_bitfield));
      if (field_type.IsValid()) {
        ConstString name;
        if (!name_sstr.empty())
          name.SetCString(name_sstr.c_str());
        sb_type_member.reset(
            new TypeMemberImpl(TypeImplSP(new TypeImpl(field_type)), bit_offset,
                               name, bitfield_bit_size, is_bitfield));
      }
    }
  }
  return LLDB_RECORD_RESULT(sb_type_member);
}

bool SBType::IsTypeComplete() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBType, IsTypeComplete);

  if (!IsValid())
    return false;
  return m_opaque_sp->GetCompilerType(false).IsCompleteType();
}

uint32_t SBType::GetTypeFlags() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBType, GetTypeFlags);

  if (!IsValid())
    return 0;
  return m_opaque_sp->GetCompilerType(true).GetTypeInfo();
}

const char *SBType::GetName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBType, GetName);

  if (!IsValid())
    return "";
  return m_opaque_sp->GetName().GetCString();
}

const char *SBType::GetDisplayTypeName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBType, GetDisplayTypeName);

  if (!IsValid())
    return "";
  return m_opaque_sp->GetDisplayTypeName().GetCString();
}

lldb::TypeClass SBType::GetTypeClass() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::TypeClass, SBType, GetTypeClass);

  if (IsValid())
    return m_opaque_sp->GetCompilerType(true).GetTypeClass();
  return lldb::eTypeClassInvalid;
}

uint32_t SBType::GetNumberOfTemplateArguments() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBType, GetNumberOfTemplateArguments);

  if (IsValid())
    return m_opaque_sp->GetCompilerType(false).GetNumTemplateArguments();
  return 0;
}

lldb::SBType SBType::GetTemplateArgumentType(uint32_t idx) {
  LLDB_RECORD_METHOD(lldb::SBType, SBType, GetTemplateArgumentType, (uint32_t),
                     idx);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBType());

  CompilerType type;
  switch(GetTemplateArgumentKind(idx)) {
    case eTemplateArgumentKindType:
      type = m_opaque_sp->GetCompilerType(false).GetTypeTemplateArgument(idx);
      break;
    case eTemplateArgumentKindIntegral:
      type = m_opaque_sp->GetCompilerType(false)
                 .GetIntegralTemplateArgument(idx)
                 ->type;
      break;
    default:
      break;
  }
  if (type.IsValid())
    return LLDB_RECORD_RESULT(SBType(type));
  return LLDB_RECORD_RESULT(SBType());
}

lldb::TemplateArgumentKind SBType::GetTemplateArgumentKind(uint32_t idx) {
  LLDB_RECORD_METHOD(lldb::TemplateArgumentKind, SBType,
                     GetTemplateArgumentKind, (uint32_t), idx);

  if (IsValid())
    return m_opaque_sp->GetCompilerType(false).GetTemplateArgumentKind(idx);
  return eTemplateArgumentKindNull;
}

SBTypeList::SBTypeList() : m_opaque_up(new TypeListImpl()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBTypeList);
}

SBTypeList::SBTypeList(const SBTypeList &rhs)
    : m_opaque_up(new TypeListImpl()) {
  LLDB_RECORD_CONSTRUCTOR(SBTypeList, (const lldb::SBTypeList &), rhs);

  for (uint32_t i = 0, rhs_size = const_cast<SBTypeList &>(rhs).GetSize();
       i < rhs_size; i++)
    Append(const_cast<SBTypeList &>(rhs).GetTypeAtIndex(i));
}

bool SBTypeList::IsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBTypeList, IsValid);
  return this->operator bool();
}
SBTypeList::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeList, operator bool);

  return (m_opaque_up != nullptr);
}

SBTypeList &SBTypeList::operator=(const SBTypeList &rhs) {
  LLDB_RECORD_METHOD(lldb::SBTypeList &,
                     SBTypeList, operator=,(const lldb::SBTypeList &), rhs);

  if (this != &rhs) {
    m_opaque_up = std::make_unique<TypeListImpl>();
    for (uint32_t i = 0, rhs_size = const_cast<SBTypeList &>(rhs).GetSize();
         i < rhs_size; i++)
      Append(const_cast<SBTypeList &>(rhs).GetTypeAtIndex(i));
  }
  return LLDB_RECORD_RESULT(*this);
}

void SBTypeList::Append(SBType type) {
  LLDB_RECORD_METHOD(void, SBTypeList, Append, (lldb::SBType), type);

  if (type.IsValid())
    m_opaque_up->Append(type.m_opaque_sp);
}

SBType SBTypeList::GetTypeAtIndex(uint32_t index) {
  LLDB_RECORD_METHOD(lldb::SBType, SBTypeList, GetTypeAtIndex, (uint32_t),
                     index);

  if (m_opaque_up)
    return LLDB_RECORD_RESULT(SBType(m_opaque_up->GetTypeAtIndex(index)));
  return LLDB_RECORD_RESULT(SBType());
}

uint32_t SBTypeList::GetSize() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBTypeList, GetSize);

  return m_opaque_up->GetSize();
}

SBTypeList::~SBTypeList() = default;

SBTypeMember::SBTypeMember() : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBTypeMember);
}

SBTypeMember::~SBTypeMember() = default;

SBTypeMember::SBTypeMember(const SBTypeMember &rhs) : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR(SBTypeMember, (const lldb::SBTypeMember &), rhs);

  if (this != &rhs) {
    if (rhs.IsValid())
      m_opaque_up = std::make_unique<TypeMemberImpl>(rhs.ref());
  }
}

lldb::SBTypeMember &SBTypeMember::operator=(const lldb::SBTypeMember &rhs) {
  LLDB_RECORD_METHOD(lldb::SBTypeMember &,
                     SBTypeMember, operator=,(const lldb::SBTypeMember &), rhs);

  if (this != &rhs) {
    if (rhs.IsValid())
      m_opaque_up = std::make_unique<TypeMemberImpl>(rhs.ref());
  }
  return LLDB_RECORD_RESULT(*this);
}

bool SBTypeMember::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeMember, IsValid);
  return this->operator bool();
}
SBTypeMember::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeMember, operator bool);

  return m_opaque_up.get();
}

const char *SBTypeMember::GetName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBTypeMember, GetName);

  if (m_opaque_up)
    return m_opaque_up->GetName().GetCString();
  return nullptr;
}

SBType SBTypeMember::GetType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBTypeMember, GetType);

  SBType sb_type;
  if (m_opaque_up) {
    sb_type.SetSP(m_opaque_up->GetTypeImpl());
  }
  return LLDB_RECORD_RESULT(sb_type);
}

uint64_t SBTypeMember::GetOffsetInBytes() {
  LLDB_RECORD_METHOD_NO_ARGS(uint64_t, SBTypeMember, GetOffsetInBytes);

  if (m_opaque_up)
    return m_opaque_up->GetBitOffset() / 8u;
  return 0;
}

uint64_t SBTypeMember::GetOffsetInBits() {
  LLDB_RECORD_METHOD_NO_ARGS(uint64_t, SBTypeMember, GetOffsetInBits);

  if (m_opaque_up)
    return m_opaque_up->GetBitOffset();
  return 0;
}

bool SBTypeMember::IsBitfield() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBTypeMember, IsBitfield);

  if (m_opaque_up)
    return m_opaque_up->GetIsBitfield();
  return false;
}

uint32_t SBTypeMember::GetBitfieldSizeInBits() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBTypeMember, GetBitfieldSizeInBits);

  if (m_opaque_up)
    return m_opaque_up->GetBitfieldBitSize();
  return 0;
}

bool SBTypeMember::GetDescription(lldb::SBStream &description,
                                  lldb::DescriptionLevel description_level) {
  LLDB_RECORD_METHOD(bool, SBTypeMember, GetDescription,
                     (lldb::SBStream &, lldb::DescriptionLevel), description,
                     description_level);

  Stream &strm = description.ref();

  if (m_opaque_up) {
    const uint32_t bit_offset = m_opaque_up->GetBitOffset();
    const uint32_t byte_offset = bit_offset / 8u;
    const uint32_t byte_bit_offset = bit_offset % 8u;
    const char *name = m_opaque_up->GetName().GetCString();
    if (byte_bit_offset)
      strm.Printf("+%u + %u bits: (", byte_offset, byte_bit_offset);
    else
      strm.Printf("+%u: (", byte_offset);

    TypeImplSP type_impl_sp(m_opaque_up->GetTypeImpl());
    if (type_impl_sp)
      type_impl_sp->GetDescription(strm, description_level);

    strm.Printf(") %s", name);
    if (m_opaque_up->GetIsBitfield()) {
      const uint32_t bitfield_bit_size = m_opaque_up->GetBitfieldBitSize();
      strm.Printf(" : %u", bitfield_bit_size);
    }
  } else {
    strm.PutCString("No value");
  }
  return true;
}

void SBTypeMember::reset(TypeMemberImpl *type_member_impl) {
  m_opaque_up.reset(type_member_impl);
}

TypeMemberImpl &SBTypeMember::ref() {
  if (m_opaque_up == nullptr)
    m_opaque_up = std::make_unique<TypeMemberImpl>();
  return *m_opaque_up;
}

const TypeMemberImpl &SBTypeMember::ref() const { return *m_opaque_up; }

SBTypeMemberFunction::SBTypeMemberFunction() : m_opaque_sp() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBTypeMemberFunction);
}

SBTypeMemberFunction::~SBTypeMemberFunction() = default;

SBTypeMemberFunction::SBTypeMemberFunction(const SBTypeMemberFunction &rhs)
    : m_opaque_sp(rhs.m_opaque_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBTypeMemberFunction,
                          (const lldb::SBTypeMemberFunction &), rhs);
}

lldb::SBTypeMemberFunction &SBTypeMemberFunction::
operator=(const lldb::SBTypeMemberFunction &rhs) {
  LLDB_RECORD_METHOD(
      lldb::SBTypeMemberFunction &,
      SBTypeMemberFunction, operator=,(const lldb::SBTypeMemberFunction &),
      rhs);

  if (this != &rhs)
    m_opaque_sp = rhs.m_opaque_sp;
  return LLDB_RECORD_RESULT(*this);
}

bool SBTypeMemberFunction::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeMemberFunction, IsValid);
  return this->operator bool();
}
SBTypeMemberFunction::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeMemberFunction, operator bool);

  return m_opaque_sp.get();
}

const char *SBTypeMemberFunction::GetName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBTypeMemberFunction, GetName);

  if (m_opaque_sp)
    return m_opaque_sp->GetName().GetCString();
  return nullptr;
}

const char *SBTypeMemberFunction::GetDemangledName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBTypeMemberFunction,
                             GetDemangledName);

  if (m_opaque_sp) {
    ConstString mangled_str = m_opaque_sp->GetMangledName();
    if (mangled_str) {
      Mangled mangled(mangled_str);
      return mangled.GetDemangledName().GetCString();
    }
  }
  return nullptr;
}

const char *SBTypeMemberFunction::GetMangledName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBTypeMemberFunction,
                             GetMangledName);

  if (m_opaque_sp)
    return m_opaque_sp->GetMangledName().GetCString();
  return nullptr;
}

SBType SBTypeMemberFunction::GetType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBTypeMemberFunction, GetType);

  SBType sb_type;
  if (m_opaque_sp) {
    sb_type.SetSP(lldb::TypeImplSP(new TypeImpl(m_opaque_sp->GetType())));
  }
  return LLDB_RECORD_RESULT(sb_type);
}

lldb::SBType SBTypeMemberFunction::GetReturnType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBTypeMemberFunction, GetReturnType);

  SBType sb_type;
  if (m_opaque_sp) {
    sb_type.SetSP(lldb::TypeImplSP(new TypeImpl(m_opaque_sp->GetReturnType())));
  }
  return LLDB_RECORD_RESULT(sb_type);
}

uint32_t SBTypeMemberFunction::GetNumberOfArguments() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBTypeMemberFunction,
                             GetNumberOfArguments);

  if (m_opaque_sp)
    return m_opaque_sp->GetNumArguments();
  return 0;
}

lldb::SBType SBTypeMemberFunction::GetArgumentTypeAtIndex(uint32_t i) {
  LLDB_RECORD_METHOD(lldb::SBType, SBTypeMemberFunction, GetArgumentTypeAtIndex,
                     (uint32_t), i);

  SBType sb_type;
  if (m_opaque_sp) {
    sb_type.SetSP(
        lldb::TypeImplSP(new TypeImpl(m_opaque_sp->GetArgumentAtIndex(i))));
  }
  return LLDB_RECORD_RESULT(sb_type);
}

lldb::MemberFunctionKind SBTypeMemberFunction::GetKind() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::MemberFunctionKind, SBTypeMemberFunction,
                             GetKind);

  if (m_opaque_sp)
    return m_opaque_sp->GetKind();
  return lldb::eMemberFunctionKindUnknown;
}

bool SBTypeMemberFunction::GetDescription(
    lldb::SBStream &description, lldb::DescriptionLevel description_level) {
  LLDB_RECORD_METHOD(bool, SBTypeMemberFunction, GetDescription,
                     (lldb::SBStream &, lldb::DescriptionLevel), description,
                     description_level);

  Stream &strm = description.ref();

  if (m_opaque_sp)
    return m_opaque_sp->GetDescription(strm);

  return false;
}

void SBTypeMemberFunction::reset(TypeMemberFunctionImpl *type_member_impl) {
  m_opaque_sp.reset(type_member_impl);
}

TypeMemberFunctionImpl &SBTypeMemberFunction::ref() {
  if (!m_opaque_sp)
    m_opaque_sp = std::make_shared<TypeMemberFunctionImpl>();
  return *m_opaque_sp.get();
}

const TypeMemberFunctionImpl &SBTypeMemberFunction::ref() const {
  return *m_opaque_sp.get();
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBType>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBType, ());
  LLDB_REGISTER_CONSTRUCTOR(SBType, (const lldb::SBType &));
  LLDB_REGISTER_METHOD(bool, SBType, operator==,(lldb::SBType &));
  LLDB_REGISTER_METHOD(bool, SBType, operator!=,(lldb::SBType &));
  LLDB_REGISTER_METHOD(lldb::SBType &,
                       SBType, operator=,(const lldb::SBType &));
  LLDB_REGISTER_METHOD_CONST(bool, SBType, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBType, operator bool, ());
  LLDB_REGISTER_METHOD(uint64_t, SBType, GetByteSize, ());
  LLDB_REGISTER_METHOD(bool, SBType, IsPointerType, ());
  LLDB_REGISTER_METHOD(bool, SBType, IsArrayType, ());
  LLDB_REGISTER_METHOD(bool, SBType, IsVectorType, ());
  LLDB_REGISTER_METHOD(bool, SBType, IsReferenceType, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetPointerType, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetPointeeType, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetReferenceType, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetTypedefedType, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetDereferencedType, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetArrayElementType, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetArrayType, (uint64_t));
  LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetVectorElementType, ());
  LLDB_REGISTER_METHOD(bool, SBType, IsFunctionType, ());
  LLDB_REGISTER_METHOD(bool, SBType, IsPolymorphicClass, ());
  LLDB_REGISTER_METHOD(bool, SBType, IsTypedefType, ());
  LLDB_REGISTER_METHOD(bool, SBType, IsAnonymousType, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetFunctionReturnType, ());
  LLDB_REGISTER_METHOD(lldb::SBTypeList, SBType, GetFunctionArgumentTypes,
                       ());
  LLDB_REGISTER_METHOD(uint32_t, SBType, GetNumberOfMemberFunctions, ());
  LLDB_REGISTER_METHOD(lldb::SBTypeMemberFunction, SBType,
                       GetMemberFunctionAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetUnqualifiedType, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetCanonicalType, ());
  LLDB_REGISTER_METHOD(lldb::BasicType, SBType, GetBasicType, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetBasicType, (lldb::BasicType));
  LLDB_REGISTER_METHOD(uint32_t, SBType, GetNumberOfDirectBaseClasses, ());
  LLDB_REGISTER_METHOD(uint32_t, SBType, GetNumberOfVirtualBaseClasses, ());
  LLDB_REGISTER_METHOD(uint32_t, SBType, GetNumberOfFields, ());
  LLDB_REGISTER_METHOD(bool, SBType, GetDescription,
                       (lldb::SBStream &, lldb::DescriptionLevel));
  LLDB_REGISTER_METHOD(lldb::SBTypeMember, SBType, GetDirectBaseClassAtIndex,
                       (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBTypeMember, SBType, GetVirtualBaseClassAtIndex,
                       (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBTypeEnumMemberList, SBType, GetEnumMembers,
                       ());
  LLDB_REGISTER_METHOD(lldb::SBTypeMember, SBType, GetFieldAtIndex,
                       (uint32_t));
  LLDB_REGISTER_METHOD(bool, SBType, IsTypeComplete, ());
  LLDB_REGISTER_METHOD(uint32_t, SBType, GetTypeFlags, ());
  LLDB_REGISTER_METHOD(const char *, SBType, GetName, ());
  LLDB_REGISTER_METHOD(const char *, SBType, GetDisplayTypeName, ());
  LLDB_REGISTER_METHOD(lldb::TypeClass, SBType, GetTypeClass, ());
  LLDB_REGISTER_METHOD(uint32_t, SBType, GetNumberOfTemplateArguments, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetTemplateArgumentType,
                       (uint32_t));
  LLDB_REGISTER_METHOD(lldb::TemplateArgumentKind, SBType,
                       GetTemplateArgumentKind, (uint32_t));
  LLDB_REGISTER_CONSTRUCTOR(SBTypeList, ());
  LLDB_REGISTER_CONSTRUCTOR(SBTypeList, (const lldb::SBTypeList &));
  LLDB_REGISTER_METHOD(bool, SBTypeList, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBTypeList, operator bool, ());
  LLDB_REGISTER_METHOD(lldb::SBTypeList &,
                       SBTypeList, operator=,(const lldb::SBTypeList &));
  LLDB_REGISTER_METHOD(void, SBTypeList, Append, (lldb::SBType));
  LLDB_REGISTER_METHOD(lldb::SBType, SBTypeList, GetTypeAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD(uint32_t, SBTypeList, GetSize, ());
  LLDB_REGISTER_CONSTRUCTOR(SBTypeMember, ());
  LLDB_REGISTER_CONSTRUCTOR(SBTypeMember, (const lldb::SBTypeMember &));
  LLDB_REGISTER_METHOD(lldb::SBTypeMember &,
                       SBTypeMember, operator=,(const lldb::SBTypeMember &));
  LLDB_REGISTER_METHOD_CONST(bool, SBTypeMember, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBTypeMember, operator bool, ());
  LLDB_REGISTER_METHOD(const char *, SBTypeMember, GetName, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBTypeMember, GetType, ());
  LLDB_REGISTER_METHOD(uint64_t, SBTypeMember, GetOffsetInBytes, ());
  LLDB_REGISTER_METHOD(uint64_t, SBTypeMember, GetOffsetInBits, ());
  LLDB_REGISTER_METHOD(bool, SBTypeMember, IsBitfield, ());
  LLDB_REGISTER_METHOD(uint32_t, SBTypeMember, GetBitfieldSizeInBits, ());
  LLDB_REGISTER_METHOD(bool, SBTypeMember, GetDescription,
                       (lldb::SBStream &, lldb::DescriptionLevel));
  LLDB_REGISTER_CONSTRUCTOR(SBTypeMemberFunction, ());
  LLDB_REGISTER_CONSTRUCTOR(SBTypeMemberFunction,
                            (const lldb::SBTypeMemberFunction &));
  LLDB_REGISTER_METHOD(
      lldb::SBTypeMemberFunction &,
      SBTypeMemberFunction, operator=,(const lldb::SBTypeMemberFunction &));
  LLDB_REGISTER_METHOD_CONST(bool, SBTypeMemberFunction, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBTypeMemberFunction, operator bool, ());
  LLDB_REGISTER_METHOD(const char *, SBTypeMemberFunction, GetName, ());
  LLDB_REGISTER_METHOD(const char *, SBTypeMemberFunction, GetDemangledName,
                       ());
  LLDB_REGISTER_METHOD(const char *, SBTypeMemberFunction, GetMangledName,
                       ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBTypeMemberFunction, GetType, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBTypeMemberFunction, GetReturnType, ());
  LLDB_REGISTER_METHOD(uint32_t, SBTypeMemberFunction, GetNumberOfArguments,
                       ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBTypeMemberFunction,
                       GetArgumentTypeAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD(lldb::MemberFunctionKind, SBTypeMemberFunction,
                       GetKind, ());
  LLDB_REGISTER_METHOD(bool, SBTypeMemberFunction, GetDescription,
                       (lldb::SBStream &, lldb::DescriptionLevel));
}

}
}
