//===--  BitcodeReader.cpp - ClangDoc Bitcode Reader ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitcodeReader.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace clang {
namespace doc {

using Record = llvm::SmallVector<uint64_t, 1024>;

// This implements decode for SmallString.
static llvm::Error decodeRecord(const Record &R,
                                llvm::SmallVectorImpl<char> &Field,
                                llvm::StringRef Blob) {
  Field.assign(Blob.begin(), Blob.end());
  return llvm::Error::success();
}

static llvm::Error decodeRecord(const Record &R, SymbolID &Field,
                                llvm::StringRef Blob) {
  if (R[0] != BitCodeConstants::USRHashSize)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "incorrect USR size");

  // First position in the record is the length of the following array, so we
  // copy the following elements to the field.
  for (int I = 0, E = R[0]; I < E; ++I)
    Field[I] = R[I + 1];
  return llvm::Error::success();
}

static llvm::Error decodeRecord(const Record &R, bool &Field,
                                llvm::StringRef Blob) {
  Field = R[0] != 0;
  return llvm::Error::success();
}

static llvm::Error decodeRecord(const Record &R, AccessSpecifier &Field,
                                llvm::StringRef Blob) {
  switch (R[0]) {
  case AS_public:
  case AS_private:
  case AS_protected:
  case AS_none:
    Field = (AccessSpecifier)R[0];
    return llvm::Error::success();
  }
  llvm_unreachable("invalid value for AccessSpecifier");
}

static llvm::Error decodeRecord(const Record &R, TagTypeKind &Field,
                                llvm::StringRef Blob) {
  switch (static_cast<TagTypeKind>(R[0])) {
  case TagTypeKind::Struct:
  case TagTypeKind::Interface:
  case TagTypeKind::Union:
  case TagTypeKind::Class:
  case TagTypeKind::Enum:
    Field = static_cast<TagTypeKind>(R[0]);
    return llvm::Error::success();
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid value for TagTypeKind");
}

static llvm::Error decodeRecord(const Record &R, std::optional<Location> &Field,
                                llvm::StringRef Blob) {
  if (R[0] > INT_MAX)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "integer too large to parse");
  Field.emplace(static_cast<int>(R[0]), static_cast<int>(R[1]), Blob,
                static_cast<bool>(R[2]));
  return llvm::Error::success();
}

static llvm::Error decodeRecord(const Record &R, InfoType &Field,
                                llvm::StringRef Blob) {
  switch (auto IT = static_cast<InfoType>(R[0])) {
  case InfoType::IT_namespace:
  case InfoType::IT_record:
  case InfoType::IT_function:
  case InfoType::IT_default:
  case InfoType::IT_enum:
  case InfoType::IT_typedef:
  case InfoType::IT_concept:
  case InfoType::IT_variable:
  case InfoType::IT_friend:
    Field = IT;
    return llvm::Error::success();
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid value for InfoType");
}

static llvm::Error decodeRecord(const Record &R, FieldId &Field,
                                llvm::StringRef Blob) {
  switch (auto F = static_cast<FieldId>(R[0])) {
  case FieldId::F_namespace:
  case FieldId::F_parent:
  case FieldId::F_vparent:
  case FieldId::F_type:
  case FieldId::F_child_namespace:
  case FieldId::F_child_record:
  case FieldId::F_concept:
  case FieldId::F_friend:
  case FieldId::F_default:
    Field = F;
    return llvm::Error::success();
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid value for FieldId");
}

static llvm::Error
decodeRecord(const Record &R,
             llvm::SmallVectorImpl<llvm::SmallString<16>> &Field,
             llvm::StringRef Blob) {
  Field.push_back(Blob);
  return llvm::Error::success();
}

static llvm::Error decodeRecord(const Record &R,
                                llvm::SmallVectorImpl<Location> &Field,
                                llvm::StringRef Blob) {
  if (R[0] > INT_MAX)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "integer too large to parse");
  Field.emplace_back(static_cast<int>(R[0]), static_cast<int>(R[1]), Blob,
                     static_cast<bool>(R[2]));
  return llvm::Error::success();
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, const unsigned VersionNo) {
  if (ID == VERSION && R[0] == VersionNo)
    return llvm::Error::success();
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mismatched bitcode version number");
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, NamespaceInfo *I) {
  switch (ID) {
  case NAMESPACE_USR:
    return decodeRecord(R, I->USR, Blob);
  case NAMESPACE_NAME:
    return decodeRecord(R, I->Name, Blob);
  case NAMESPACE_PATH:
    return decodeRecord(R, I->Path, Blob);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid field for NamespaceInfo");
  }
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, RecordInfo *I) {
  switch (ID) {
  case RECORD_USR:
    return decodeRecord(R, I->USR, Blob);
  case RECORD_NAME:
    return decodeRecord(R, I->Name, Blob);
  case RECORD_PATH:
    return decodeRecord(R, I->Path, Blob);
  case RECORD_DEFLOCATION:
    return decodeRecord(R, I->DefLoc, Blob);
  case RECORD_LOCATION:
    return decodeRecord(R, I->Loc, Blob);
  case RECORD_TAG_TYPE:
    return decodeRecord(R, I->TagType, Blob);
  case RECORD_IS_TYPE_DEF:
    return decodeRecord(R, I->IsTypeDef, Blob);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid field for RecordInfo");
  }
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, BaseRecordInfo *I) {
  switch (ID) {
  case BASE_RECORD_USR:
    return decodeRecord(R, I->USR, Blob);
  case BASE_RECORD_NAME:
    return decodeRecord(R, I->Name, Blob);
  case BASE_RECORD_PATH:
    return decodeRecord(R, I->Path, Blob);
  case BASE_RECORD_TAG_TYPE:
    return decodeRecord(R, I->TagType, Blob);
  case BASE_RECORD_IS_VIRTUAL:
    return decodeRecord(R, I->IsVirtual, Blob);
  case BASE_RECORD_ACCESS:
    return decodeRecord(R, I->Access, Blob);
  case BASE_RECORD_IS_PARENT:
    return decodeRecord(R, I->IsParent, Blob);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid field for BaseRecordInfo");
  }
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, EnumInfo *I) {
  switch (ID) {
  case ENUM_USR:
    return decodeRecord(R, I->USR, Blob);
  case ENUM_NAME:
    return decodeRecord(R, I->Name, Blob);
  case ENUM_DEFLOCATION:
    return decodeRecord(R, I->DefLoc, Blob);
  case ENUM_LOCATION:
    return decodeRecord(R, I->Loc, Blob);
  case ENUM_SCOPED:
    return decodeRecord(R, I->Scoped, Blob);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid field for EnumInfo");
  }
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, TypedefInfo *I) {
  switch (ID) {
  case TYPEDEF_USR:
    return decodeRecord(R, I->USR, Blob);
  case TYPEDEF_NAME:
    return decodeRecord(R, I->Name, Blob);
  case TYPEDEF_DEFLOCATION:
    return decodeRecord(R, I->DefLoc, Blob);
  case TYPEDEF_IS_USING:
    return decodeRecord(R, I->IsUsing, Blob);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid field for TypedefInfo");
  }
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, EnumValueInfo *I) {
  switch (ID) {
  case ENUM_VALUE_NAME:
    return decodeRecord(R, I->Name, Blob);
  case ENUM_VALUE_VALUE:
    return decodeRecord(R, I->Value, Blob);
  case ENUM_VALUE_EXPR:
    return decodeRecord(R, I->ValueExpr, Blob);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid field for EnumValueInfo");
  }
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, FunctionInfo *I) {
  switch (ID) {
  case FUNCTION_USR:
    return decodeRecord(R, I->USR, Blob);
  case FUNCTION_NAME:
    return decodeRecord(R, I->Name, Blob);
  case FUNCTION_DEFLOCATION:
    return decodeRecord(R, I->DefLoc, Blob);
  case FUNCTION_LOCATION:
    return decodeRecord(R, I->Loc, Blob);
  case FUNCTION_ACCESS:
    return decodeRecord(R, I->Access, Blob);
  case FUNCTION_IS_METHOD:
    return decodeRecord(R, I->IsMethod, Blob);
  case FUNCTION_IS_STATIC:
    return decodeRecord(R, I->IsStatic, Blob);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid field for FunctionInfo");
  }
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, TypeInfo *I) {
  switch (ID) {
  case TYPE_IS_BUILTIN:
    return decodeRecord(R, I->IsBuiltIn, Blob);
  case TYPE_IS_TEMPLATE:
    return decodeRecord(R, I->IsTemplate, Blob);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid field for TypeInfo");
  }
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, FieldTypeInfo *I) {
  switch (ID) {
  case FIELD_TYPE_NAME:
    return decodeRecord(R, I->Name, Blob);
  case FIELD_DEFAULT_VALUE:
    return decodeRecord(R, I->DefaultValue, Blob);
  case FIELD_TYPE_IS_BUILTIN:
    return decodeRecord(R, I->IsBuiltIn, Blob);
  case FIELD_TYPE_IS_TEMPLATE:
    return decodeRecord(R, I->IsTemplate, Blob);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid field for TypeInfo");
  }
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, MemberTypeInfo *I) {
  switch (ID) {
  case MEMBER_TYPE_NAME:
    return decodeRecord(R, I->Name, Blob);
  case MEMBER_TYPE_ACCESS:
    return decodeRecord(R, I->Access, Blob);
  case MEMBER_TYPE_IS_STATIC:
    return decodeRecord(R, I->IsStatic, Blob);
  case MEMBER_TYPE_IS_BUILTIN:
    return decodeRecord(R, I->IsBuiltIn, Blob);
  case MEMBER_TYPE_IS_TEMPLATE:
    return decodeRecord(R, I->IsTemplate, Blob);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid field for MemberTypeInfo");
  }
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, CommentInfo *I) {
  llvm::SmallString<16> KindStr;
  switch (ID) {
  case COMMENT_KIND:
    if (llvm::Error Err = decodeRecord(R, KindStr, Blob))
      return Err;
    I->Kind = stringToCommentKind(KindStr);
    return llvm::Error::success();
  case COMMENT_TEXT:
    return decodeRecord(R, I->Text, Blob);
  case COMMENT_NAME:
    return decodeRecord(R, I->Name, Blob);
  case COMMENT_DIRECTION:
    return decodeRecord(R, I->Direction, Blob);
  case COMMENT_PARAMNAME:
    return decodeRecord(R, I->ParamName, Blob);
  case COMMENT_CLOSENAME:
    return decodeRecord(R, I->CloseName, Blob);
  case COMMENT_ATTRKEY:
    return decodeRecord(R, I->AttrKeys, Blob);
  case COMMENT_ATTRVAL:
    return decodeRecord(R, I->AttrValues, Blob);
  case COMMENT_ARG:
    return decodeRecord(R, I->Args, Blob);
  case COMMENT_SELFCLOSING:
    return decodeRecord(R, I->SelfClosing, Blob);
  case COMMENT_EXPLICIT:
    return decodeRecord(R, I->Explicit, Blob);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid field for CommentInfo");
  }
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, Reference *I, FieldId &F) {
  switch (ID) {
  case REFERENCE_USR:
    return decodeRecord(R, I->USR, Blob);
  case REFERENCE_NAME:
    return decodeRecord(R, I->Name, Blob);
  case REFERENCE_QUAL_NAME:
    return decodeRecord(R, I->QualName, Blob);
  case REFERENCE_TYPE:
    return decodeRecord(R, I->RefType, Blob);
  case REFERENCE_PATH:
    return decodeRecord(R, I->Path, Blob);
  case REFERENCE_FIELD:
    return decodeRecord(R, F, Blob);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid field for Reference");
  }
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, TemplateInfo *I) {
  // Currently there are no child records of TemplateInfo (only child blocks).
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid field for TemplateParamInfo");
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob,
                               TemplateSpecializationInfo *I) {
  if (ID == TEMPLATE_SPECIALIZATION_OF)
    return decodeRecord(R, I->SpecializationOf, Blob);
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid field for TemplateParamInfo");
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, TemplateParamInfo *I) {
  if (ID == TEMPLATE_PARAM_CONTENTS)
    return decodeRecord(R, I->Contents, Blob);
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid field for TemplateParamInfo");
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, ConceptInfo *I) {
  switch (ID) {
  case CONCEPT_USR:
    return decodeRecord(R, I->USR, Blob);
  case CONCEPT_NAME:
    return decodeRecord(R, I->Name, Blob);
  case CONCEPT_IS_TYPE:
    return decodeRecord(R, I->IsType, Blob);
  case CONCEPT_CONSTRAINT_EXPRESSION:
    return decodeRecord(R, I->ConstraintExpression, Blob);
  }
  llvm_unreachable("invalid field for ConceptInfo");
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, ConstraintInfo *I) {
  if (ID == CONSTRAINT_EXPRESSION)
    return decodeRecord(R, I->ConstraintExpr, Blob);
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid field for ConstraintInfo");
}

static llvm::Error parseRecord(const Record &R, unsigned ID,
                               llvm::StringRef Blob, VarInfo *I) {
  switch (ID) {
  case VAR_USR:
    return decodeRecord(R, I->USR, Blob);
  case VAR_NAME:
    return decodeRecord(R, I->Name, Blob);
  case VAR_DEFLOCATION:
    return decodeRecord(R, I->DefLoc, Blob);
  case VAR_IS_STATIC:
    return decodeRecord(R, I->IsStatic, Blob);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid field for VarInfo");
  }
}

static llvm::Error parseRecord(const Record &R, unsigned ID, StringRef Blob,
                               FriendInfo *F) {
  if (ID == FRIEND_IS_CLASS) {
    return decodeRecord(R, F->IsClass, Blob);
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid field for Friend");
}

template <typename T> static llvm::Expected<CommentInfo *> getCommentInfo(T I) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid type cannot contain CommentInfo");
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(FunctionInfo *I) {
  return &I->Description.emplace_back();
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(NamespaceInfo *I) {
  return &I->Description.emplace_back();
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(RecordInfo *I) {
  return &I->Description.emplace_back();
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(MemberTypeInfo *I) {
  return &I->Description.emplace_back();
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(EnumInfo *I) {
  return &I->Description.emplace_back();
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(TypedefInfo *I) {
  return &I->Description.emplace_back();
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(EnumValueInfo *I) {
  return &I->Description.emplace_back();
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(CommentInfo *I) {
  I->Children.emplace_back(std::make_unique<CommentInfo>());
  return I->Children.back().get();
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(ConceptInfo *I) {
  return &I->Description.emplace_back();
}

template <> Expected<CommentInfo *> getCommentInfo(VarInfo *I) {
  return &I->Description.emplace_back();
}

// When readSubBlock encounters a TypeInfo sub-block, it calls addTypeInfo on
// the parent block to set it. The template specializations define what to do
// for each supported parent block.
template <typename T, typename TTypeInfo>
static llvm::Error addTypeInfo(T I, TTypeInfo &&TI) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid type cannot contain TypeInfo");
}

template <> llvm::Error addTypeInfo(RecordInfo *I, MemberTypeInfo &&T) {
  I->Members.emplace_back(std::move(T));
  return llvm::Error::success();
}

template <> llvm::Error addTypeInfo(BaseRecordInfo *I, MemberTypeInfo &&T) {
  I->Members.emplace_back(std::move(T));
  return llvm::Error::success();
}

template <> llvm::Error addTypeInfo(FunctionInfo *I, TypeInfo &&T) {
  I->ReturnType = std::move(T);
  return llvm::Error::success();
}

template <> llvm::Error addTypeInfo(FunctionInfo *I, FieldTypeInfo &&T) {
  I->Params.emplace_back(std::move(T));
  return llvm::Error::success();
}

template <> llvm::Error addTypeInfo(FriendInfo *I, FieldTypeInfo &&T) {
  if (!I->Params)
    I->Params.emplace();
  I->Params->emplace_back(std::move(T));
  return llvm::Error::success();
}

template <> llvm::Error addTypeInfo(FriendInfo *I, TypeInfo &&T) {
  I->ReturnType.emplace(std::move(T));
  return llvm::Error::success();
}

template <> llvm::Error addTypeInfo(EnumInfo *I, TypeInfo &&T) {
  I->BaseType = std::move(T);
  return llvm::Error::success();
}

template <> llvm::Error addTypeInfo(TypedefInfo *I, TypeInfo &&T) {
  I->Underlying = std::move(T);
  return llvm::Error::success();
}

template <> llvm::Error addTypeInfo(VarInfo *I, TypeInfo &&T) {
  I->Type = std::move(T);
  return llvm::Error::success();
}

template <typename T>
static llvm::Error addReference(T I, Reference &&R, FieldId F) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid type cannot contain Reference");
}

template <> llvm::Error addReference(VarInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_namespace:
    I->Namespace.emplace_back(std::move(R));
    return llvm::Error::success();
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "VarInfo cannot contain this Reference");
  }
}

template <> llvm::Error addReference(TypeInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_type:
    I->Type = std::move(R);
    return llvm::Error::success();
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid type cannot contain Reference");
  }
}

template <>
llvm::Error addReference(FieldTypeInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_type:
    I->Type = std::move(R);
    return llvm::Error::success();
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid type cannot contain Reference");
  }
}

template <>
llvm::Error addReference(MemberTypeInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_type:
    I->Type = std::move(R);
    return llvm::Error::success();
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid type cannot contain Reference");
  }
}

template <> llvm::Error addReference(EnumInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_namespace:
    I->Namespace.emplace_back(std::move(R));
    return llvm::Error::success();
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid type cannot contain Reference");
  }
}

template <> llvm::Error addReference(TypedefInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_namespace:
    I->Namespace.emplace_back(std::move(R));
    return llvm::Error::success();
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid type cannot contain Reference");
  }
}

template <>
llvm::Error addReference(NamespaceInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_namespace:
    I->Namespace.emplace_back(std::move(R));
    return llvm::Error::success();
  case FieldId::F_child_namespace:
    I->Children.Namespaces.emplace_back(std::move(R));
    return llvm::Error::success();
  case FieldId::F_child_record:
    I->Children.Records.emplace_back(std::move(R));
    return llvm::Error::success();
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid type cannot contain Reference");
  }
}

template <>
llvm::Error addReference(FunctionInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_namespace:
    I->Namespace.emplace_back(std::move(R));
    return llvm::Error::success();
  case FieldId::F_parent:
    I->Parent = std::move(R);
    return llvm::Error::success();
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid type cannot contain Reference");
  }
}

template <> llvm::Error addReference(RecordInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_namespace:
    I->Namespace.emplace_back(std::move(R));
    return llvm::Error::success();
  case FieldId::F_parent:
    I->Parents.emplace_back(std::move(R));
    return llvm::Error::success();
  case FieldId::F_vparent:
    I->VirtualParents.emplace_back(std::move(R));
    return llvm::Error::success();
  case FieldId::F_child_record:
    I->Children.Records.emplace_back(std::move(R));
    return llvm::Error::success();
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid type cannot contain Reference");
  }
}

template <>
llvm::Error addReference(ConstraintInfo *I, Reference &&R, FieldId F) {
  if (F == FieldId::F_concept) {
    I->ConceptRef = std::move(R);
    return llvm::Error::success();
  }
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "ConstraintInfo cannot contain this Reference");
}

template <>
llvm::Error addReference(FriendInfo *Friend, Reference &&R, FieldId F) {
  if (F == FieldId::F_friend) {
    Friend->Ref = std::move(R);
    return llvm::Error::success();
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Friend cannot contain this Reference");
}

template <typename T, typename ChildInfoType>
static void addChild(T I, ChildInfoType &&R) {
  llvm::errs() << "invalid child type for info";
  exit(1);
}

// Namespace children:
template <> void addChild(NamespaceInfo *I, FunctionInfo &&R) {
  I->Children.Functions.emplace_back(std::move(R));
}
template <> void addChild(NamespaceInfo *I, EnumInfo &&R) {
  I->Children.Enums.emplace_back(std::move(R));
}
template <> void addChild(NamespaceInfo *I, TypedefInfo &&R) {
  I->Children.Typedefs.emplace_back(std::move(R));
}
template <> void addChild(NamespaceInfo *I, ConceptInfo &&R) {
  I->Children.Concepts.emplace_back(std::move(R));
}
template <> void addChild(NamespaceInfo *I, VarInfo &&R) {
  I->Children.Variables.emplace_back(std::move(R));
}

// Record children:
template <> void addChild(RecordInfo *I, FunctionInfo &&R) {
  I->Children.Functions.emplace_back(std::move(R));
}
template <> void addChild(RecordInfo *I, EnumInfo &&R) {
  I->Children.Enums.emplace_back(std::move(R));
}
template <> void addChild(RecordInfo *I, TypedefInfo &&R) {
  I->Children.Typedefs.emplace_back(std::move(R));
}
template <> void addChild(RecordInfo *I, FriendInfo &&R) {
  I->Friends.emplace_back(std::move(R));
}

// Other types of children:
template <> void addChild(EnumInfo *I, EnumValueInfo &&R) {
  I->Members.emplace_back(std::move(R));
}
template <> void addChild(RecordInfo *I, BaseRecordInfo &&R) {
  I->Bases.emplace_back(std::move(R));
}
template <> void addChild(BaseRecordInfo *I, FunctionInfo &&R) {
  I->Children.Functions.emplace_back(std::move(R));
}

// TemplateParam children. These go into either a TemplateInfo (for template
// parameters) or TemplateSpecializationInfo (for the specialization's
// parameters).
template <typename T> static void addTemplateParam(T I, TemplateParamInfo &&P) {
  llvm::errs() << "invalid container for template parameter";
  exit(1);
}
template <> void addTemplateParam(TemplateInfo *I, TemplateParamInfo &&P) {
  I->Params.emplace_back(std::move(P));
}
template <>
void addTemplateParam(TemplateSpecializationInfo *I, TemplateParamInfo &&P) {
  I->Params.emplace_back(std::move(P));
}

// Template info. These apply to either records or functions.
template <typename T> static void addTemplate(T I, TemplateInfo &&P) {
  llvm::errs() << "invalid container for template info";
  exit(1);
}
template <> void addTemplate(RecordInfo *I, TemplateInfo &&P) {
  I->Template.emplace(std::move(P));
}
template <> void addTemplate(FunctionInfo *I, TemplateInfo &&P) {
  I->Template.emplace(std::move(P));
}
template <> void addTemplate(ConceptInfo *I, TemplateInfo &&P) {
  I->Template = std::move(P);
}
template <> void addTemplate(FriendInfo *I, TemplateInfo &&P) {
  I->Template.emplace(std::move(P));
}

// Template specializations go only into template records.
template <typename T>
static void addTemplateSpecialization(T I, TemplateSpecializationInfo &&TSI) {
  llvm::errs() << "invalid container for template specialization info";
  exit(1);
}
template <>
void addTemplateSpecialization(TemplateInfo *I,
                               TemplateSpecializationInfo &&TSI) {
  I->Specialization.emplace(std::move(TSI));
}

template <typename T> static void addConstraint(T I, ConstraintInfo &&C) {
  llvm::errs() << "invalid container for constraint info";
  exit(1);
}
template <> void addConstraint(TemplateInfo *I, ConstraintInfo &&C) {
  I->Constraints.emplace_back(std::move(C));
}

// Read records from bitcode into a given info.
template <typename T>
llvm::Error ClangDocBitcodeReader::readRecord(unsigned ID, T I) {
  Record R;
  llvm::StringRef Blob;
  llvm::Expected<unsigned> MaybeRecID = Stream.readRecord(ID, R, &Blob);
  if (!MaybeRecID)
    return MaybeRecID.takeError();
  return parseRecord(R, MaybeRecID.get(), Blob, I);
}

template <>
llvm::Error ClangDocBitcodeReader::readRecord(unsigned ID, Reference *I) {
  llvm::TimeTraceScope("Reducing infos", "readRecord");
  Record R;
  llvm::StringRef Blob;
  llvm::Expected<unsigned> MaybeRecID = Stream.readRecord(ID, R, &Blob);
  if (!MaybeRecID)
    return MaybeRecID.takeError();
  return parseRecord(R, MaybeRecID.get(), Blob, I, CurrentReferenceField);
}

// Read a block of records into a single info.
template <typename T>
llvm::Error ClangDocBitcodeReader::readBlock(unsigned ID, T I) {
  llvm::TimeTraceScope("Reducing infos", "readBlock");
  if (llvm::Error Err = Stream.EnterSubBlock(ID))
    return Err;

  while (true) {
    unsigned BlockOrCode = 0;
    Cursor Res = skipUntilRecordOrBlock(BlockOrCode);

    switch (Res) {
    case Cursor::BadBlock:
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "bad block found");
    case Cursor::BlockEnd:
      return llvm::Error::success();
    case Cursor::BlockBegin:
      if (llvm::Error Err = readSubBlock(BlockOrCode, I)) {
        if (llvm::Error Skipped = Stream.SkipBlock())
          return joinErrors(std::move(Err), std::move(Skipped));
        return Err;
      }
      continue;
    case Cursor::Record:
      break;
    }
    if (auto Err = readRecord(BlockOrCode, I))
      return Err;
  }
}

// TODO: fix inconsistentent returning of errors in add callbacks.
// Once that's fixed, we only need one handleSubBlock.
template <typename InfoType, typename T, typename Callback>
llvm::Error ClangDocBitcodeReader::handleSubBlock(unsigned ID, T Parent,
                                                  Callback Function) {
  InfoType Info;
  if (auto Err = readBlock(ID, &Info))
    return Err;
  Function(Parent, std::move(Info));
  return llvm::Error::success();
}

template <typename InfoType, typename T, typename Callback>
llvm::Error ClangDocBitcodeReader::handleTypeSubBlock(unsigned ID, T Parent,
                                                      Callback Function) {
  InfoType Info;
  if (auto Err = readBlock(ID, &Info))
    return Err;
  if (auto Err = Function(Parent, std::move(Info)))
    return Err;
  return llvm::Error::success();
}

template <typename T>
llvm::Error ClangDocBitcodeReader::readSubBlock(unsigned ID, T I) {
  llvm::TimeTraceScope("Reducing infos", "readSubBlock");

  static auto CreateAddFunc = [](auto AddFunc) {
    return [AddFunc](auto Parent, auto Child) {
      return AddFunc(Parent, std::move(Child));
    };
  };

  switch (ID) {
  // Blocks can only have certain types of sub blocks.
  case BI_COMMENT_BLOCK_ID: {
    auto Comment = getCommentInfo(I);
    if (!Comment)
      return Comment.takeError();
    if (auto Err = readBlock(ID, Comment.get()))
      return Err;
    return llvm::Error::success();
  }
  case BI_TYPE_BLOCK_ID: {
    return handleTypeSubBlock<TypeInfo>(
        ID, I, CreateAddFunc(addTypeInfo<T, TypeInfo>));
  }
  case BI_FIELD_TYPE_BLOCK_ID: {
    return handleTypeSubBlock<FieldTypeInfo>(
        ID, I, CreateAddFunc(addTypeInfo<T, FieldTypeInfo>));
  }
  case BI_MEMBER_TYPE_BLOCK_ID: {
    return handleTypeSubBlock<MemberTypeInfo>(
        ID, I, CreateAddFunc(addTypeInfo<T, MemberTypeInfo>));
  }
  case BI_REFERENCE_BLOCK_ID: {
    Reference R;
    if (auto Err = readBlock(ID, &R))
      return Err;
    if (auto Err = addReference(I, std::move(R), CurrentReferenceField))
      return Err;
    return llvm::Error::success();
  }
  case BI_FUNCTION_BLOCK_ID: {
    return handleSubBlock<FunctionInfo>(
        ID, I, CreateAddFunc(addChild<T, FunctionInfo>));
  }
  case BI_BASE_RECORD_BLOCK_ID: {
    return handleSubBlock<BaseRecordInfo>(
        ID, I, CreateAddFunc(addChild<T, BaseRecordInfo>));
  }
  case BI_ENUM_BLOCK_ID: {
    return handleSubBlock<EnumInfo>(ID, I,
                                    CreateAddFunc(addChild<T, EnumInfo>));
  }
  case BI_ENUM_VALUE_BLOCK_ID: {
    return handleSubBlock<EnumValueInfo>(
        ID, I, CreateAddFunc(addChild<T, EnumValueInfo>));
  }
  case BI_TEMPLATE_BLOCK_ID: {
    return handleSubBlock<TemplateInfo>(ID, I, CreateAddFunc(addTemplate<T>));
  }
  case BI_TEMPLATE_SPECIALIZATION_BLOCK_ID: {
    return handleSubBlock<TemplateSpecializationInfo>(
        ID, I, CreateAddFunc(addTemplateSpecialization<T>));
  }
  case BI_TEMPLATE_PARAM_BLOCK_ID: {
    return handleSubBlock<TemplateParamInfo>(
        ID, I, CreateAddFunc(addTemplateParam<T>));
  }
  case BI_TYPEDEF_BLOCK_ID: {
    return handleSubBlock<TypedefInfo>(ID, I,
                                       CreateAddFunc(addChild<T, TypedefInfo>));
  }
  case BI_CONSTRAINT_BLOCK_ID: {
    return handleSubBlock<ConstraintInfo>(ID, I,
                                          CreateAddFunc(addConstraint<T>));
  }
  case BI_CONCEPT_BLOCK_ID: {
    return handleSubBlock<ConceptInfo>(ID, I,
                                       CreateAddFunc(addChild<T, ConceptInfo>));
  }
  case BI_VAR_BLOCK_ID: {
    return handleSubBlock<VarInfo>(ID, I, CreateAddFunc(addChild<T, VarInfo>));
  }
  case BI_FRIEND_BLOCK_ID: {
    return handleSubBlock<FriendInfo>(ID, I,
                                      CreateAddFunc(addChild<T, FriendInfo>));
  }
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid subblock type");
  }
}

ClangDocBitcodeReader::Cursor
ClangDocBitcodeReader::skipUntilRecordOrBlock(unsigned &BlockOrRecordID) {
  llvm::TimeTraceScope("Reducing infos", "skipUntilRecordOrBlock");
  BlockOrRecordID = 0;

  while (!Stream.AtEndOfStream()) {
    Expected<unsigned> MaybeCode = Stream.ReadCode();
    if (!MaybeCode) {
      // FIXME this drops the error on the floor.
      consumeError(MaybeCode.takeError());
      return Cursor::BadBlock;
    }

    unsigned Code = MaybeCode.get();
    if (Code >= static_cast<unsigned>(llvm::bitc::FIRST_APPLICATION_ABBREV)) {
      BlockOrRecordID = Code;
      return Cursor::Record;
    }
    switch (static_cast<llvm::bitc::FixedAbbrevIDs>(Code)) {
    case llvm::bitc::ENTER_SUBBLOCK:
      if (Expected<unsigned> MaybeID = Stream.ReadSubBlockID())
        BlockOrRecordID = MaybeID.get();
      else {
        // FIXME this drops the error on the floor.
        consumeError(MaybeID.takeError());
      }
      return Cursor::BlockBegin;
    case llvm::bitc::END_BLOCK:
      if (Stream.ReadBlockEnd())
        return Cursor::BadBlock;
      return Cursor::BlockEnd;
    case llvm::bitc::DEFINE_ABBREV:
      if (llvm::Error Err = Stream.ReadAbbrevRecord()) {
        // FIXME this drops the error on the floor.
        consumeError(std::move(Err));
      }
      continue;
    case llvm::bitc::UNABBREV_RECORD:
      return Cursor::BadBlock;
    case llvm::bitc::FIRST_APPLICATION_ABBREV:
      llvm_unreachable("Unexpected abbrev id.");
    }
  }
  llvm_unreachable("Premature stream end.");
}

llvm::Error ClangDocBitcodeReader::validateStream() {
  if (Stream.AtEndOfStream())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "premature end of stream");

  // Sniff for the signature.
  for (int Idx = 0; Idx != 4; ++Idx) {
    Expected<llvm::SimpleBitstreamCursor::word_t> MaybeRead = Stream.Read(8);
    if (!MaybeRead)
      return MaybeRead.takeError();
    if (MaybeRead.get() != BitCodeConstants::Signature[Idx])
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid bitcode signature");
  }
  return llvm::Error::success();
}

llvm::Error ClangDocBitcodeReader::readBlockInfoBlock() {
  llvm::TimeTraceScope("Reducing infos", "readBlockInfoBlock");
  Expected<std::optional<llvm::BitstreamBlockInfo>> MaybeBlockInfo =
      Stream.ReadBlockInfoBlock();
  if (!MaybeBlockInfo)
    return MaybeBlockInfo.takeError();
  BlockInfo = MaybeBlockInfo.get();
  if (!BlockInfo)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unable to parse BlockInfoBlock");
  Stream.setBlockInfo(&*BlockInfo);
  return llvm::Error::success();
}

template <typename T>
llvm::Expected<std::unique_ptr<Info>>
ClangDocBitcodeReader::createInfo(unsigned ID) {
  llvm::TimeTraceScope("Reducing infos", "createInfo");
  std::unique_ptr<Info> I = std::make_unique<T>();
  if (auto Err = readBlock(ID, static_cast<T *>(I.get())))
    return std::move(Err);
  return std::unique_ptr<Info>{std::move(I)};
}

llvm::Expected<std::unique_ptr<Info>>
ClangDocBitcodeReader::readBlockToInfo(unsigned ID) {
  llvm::TimeTraceScope("Reducing infos", "readBlockToInfo");
  switch (ID) {
  case BI_NAMESPACE_BLOCK_ID:
    return createInfo<NamespaceInfo>(ID);
  case BI_RECORD_BLOCK_ID:
    return createInfo<RecordInfo>(ID);
  case BI_ENUM_BLOCK_ID:
    return createInfo<EnumInfo>(ID);
  case BI_TYPEDEF_BLOCK_ID:
    return createInfo<TypedefInfo>(ID);
  case BI_CONCEPT_BLOCK_ID:
    return createInfo<ConceptInfo>(ID);
  case BI_FUNCTION_BLOCK_ID:
    return createInfo<FunctionInfo>(ID);
  case BI_VAR_BLOCK_ID:
    return createInfo<VarInfo>(ID);
  case BI_FRIEND_BLOCK_ID:
    return createInfo<FriendInfo>(ID);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot create info");
  }
}

// Entry point
llvm::Expected<std::vector<std::unique_ptr<Info>>>
ClangDocBitcodeReader::readBitcode() {
  std::vector<std::unique_ptr<Info>> Infos;
  if (auto Err = validateStream())
    return std::move(Err);

  // Read the top level blocks.
  while (!Stream.AtEndOfStream()) {
    Expected<unsigned> MaybeCode = Stream.ReadCode();
    if (!MaybeCode)
      return MaybeCode.takeError();
    if (MaybeCode.get() != llvm::bitc::ENTER_SUBBLOCK)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "no blocks in input");
    Expected<unsigned> MaybeID = Stream.ReadSubBlockID();
    if (!MaybeID)
      return MaybeID.takeError();
    unsigned ID = MaybeID.get();
    switch (ID) {
    // NamedType and Comment blocks should not appear at the top level
    case BI_TYPE_BLOCK_ID:
    case BI_FIELD_TYPE_BLOCK_ID:
    case BI_MEMBER_TYPE_BLOCK_ID:
    case BI_COMMENT_BLOCK_ID:
    case BI_REFERENCE_BLOCK_ID:
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid top level block");
    case BI_NAMESPACE_BLOCK_ID:
    case BI_RECORD_BLOCK_ID:
    case BI_ENUM_BLOCK_ID:
    case BI_TYPEDEF_BLOCK_ID:
    case BI_CONCEPT_BLOCK_ID:
    case BI_VAR_BLOCK_ID:
    case BI_FRIEND_BLOCK_ID:
    case BI_FUNCTION_BLOCK_ID: {
      auto InfoOrErr = readBlockToInfo(ID);
      if (!InfoOrErr)
        return InfoOrErr.takeError();
      Infos.emplace_back(std::move(InfoOrErr.get()));
      continue;
    }
    case BI_VERSION_BLOCK_ID:
      if (auto Err = readBlock(ID, VersionNumber))
        return std::move(Err);
      continue;
    case llvm::bitc::BLOCKINFO_BLOCK_ID:
      if (auto Err = readBlockInfoBlock())
        return std::move(Err);
      continue;
    default:
      if (llvm::Error Err = Stream.SkipBlock()) {
        // FIXME this drops the error on the floor.
        consumeError(std::move(Err));
      }
      continue;
    }
  }
  return std::move(Infos);
}

} // namespace doc
} // namespace clang
