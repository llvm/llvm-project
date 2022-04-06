//===- ExtractAPI/API.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the APIRecord and derived record structs,
/// and the APISet class.
///
//===----------------------------------------------------------------------===//

#include "clang/ExtractAPI/API.h"
#include "clang/AST/CommentCommandTraits.h"
#include "clang/AST/CommentLexer.h"
#include "clang/AST/RawCommentList.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

using namespace clang::extractapi;
using namespace llvm;

namespace {

template <typename RecordTy, typename... CtorArgsTy>
RecordTy *addTopLevelRecord(APISet::RecordMap<RecordTy> &RecordMap,
                            StringRef Name, CtorArgsTy &&...CtorArgs) {
  auto Result = RecordMap.insert({Name, nullptr});

  // Create the record if it does not already exist
  if (Result.second)
    Result.first->second =
        std::make_unique<RecordTy>(Name, std::forward<CtorArgsTy>(CtorArgs)...);

  return Result.first->second.get();
}

} // namespace

GlobalRecord *APISet::addGlobal(GVKind Kind, StringRef Name, StringRef USR,
                                PresumedLoc Loc,
                                const AvailabilityInfo &Availability,
                                LinkageInfo Linkage, const DocComment &Comment,
                                DeclarationFragments Fragments,
                                DeclarationFragments SubHeading,
                                FunctionSignature Signature) {
  return addTopLevelRecord(Globals, Name, USR, Loc, Availability, Linkage,
                           Comment, Fragments, SubHeading, Kind, Signature);
}

GlobalRecord *
APISet::addGlobalVar(StringRef Name, StringRef USR, PresumedLoc Loc,
                     const AvailabilityInfo &Availability, LinkageInfo Linkage,
                     const DocComment &Comment, DeclarationFragments Fragments,
                     DeclarationFragments SubHeading) {
  return addGlobal(GVKind::Variable, Name, USR, Loc, Availability, Linkage,
                   Comment, Fragments, SubHeading, {});
}

GlobalRecord *
APISet::addFunction(StringRef Name, StringRef USR, PresumedLoc Loc,
                    const AvailabilityInfo &Availability, LinkageInfo Linkage,
                    const DocComment &Comment, DeclarationFragments Fragments,
                    DeclarationFragments SubHeading,
                    FunctionSignature Signature) {
  return addGlobal(GVKind::Function, Name, USR, Loc, Availability, Linkage,
                   Comment, Fragments, SubHeading, Signature);
}

EnumConstantRecord *APISet::addEnumConstant(
    EnumRecord *Enum, StringRef Name, StringRef USR, PresumedLoc Loc,
    const AvailabilityInfo &Availability, const DocComment &Comment,
    DeclarationFragments Declaration, DeclarationFragments SubHeading) {
  auto Record = std::make_unique<EnumConstantRecord>(
      Name, USR, Loc, Availability, Comment, Declaration, SubHeading);
  return Enum->Constants.emplace_back(std::move(Record)).get();
}

EnumRecord *APISet::addEnum(StringRef Name, StringRef USR, PresumedLoc Loc,
                            const AvailabilityInfo &Availability,
                            const DocComment &Comment,
                            DeclarationFragments Declaration,
                            DeclarationFragments SubHeading) {
  return addTopLevelRecord(Enums, Name, USR, Loc, Availability, Comment,
                           Declaration, SubHeading);
}

StructFieldRecord *APISet::addStructField(StructRecord *Struct, StringRef Name,
                                          StringRef USR, PresumedLoc Loc,
                                          const AvailabilityInfo &Availability,
                                          const DocComment &Comment,
                                          DeclarationFragments Declaration,
                                          DeclarationFragments SubHeading) {
  auto Record = std::make_unique<StructFieldRecord>(
      Name, USR, Loc, Availability, Comment, Declaration, SubHeading);
  return Struct->Fields.emplace_back(std::move(Record)).get();
}

StructRecord *APISet::addStruct(StringRef Name, StringRef USR, PresumedLoc Loc,
                                const AvailabilityInfo &Availability,
                                const DocComment &Comment,
                                DeclarationFragments Declaration,
                                DeclarationFragments SubHeading) {
  return addTopLevelRecord(Structs, Name, USR, Loc, Availability, Comment,
                           Declaration, SubHeading);
}

ObjCInterfaceRecord *APISet::addObjCInterface(
    StringRef Name, StringRef USR, PresumedLoc Loc,
    const AvailabilityInfo &Availability, LinkageInfo Linkage,
    const DocComment &Comment, DeclarationFragments Declaration,
    DeclarationFragments SubHeading, SymbolReference SuperClass) {
  return addTopLevelRecord(ObjCInterfaces, Name, USR, Loc, Availability,
                           Linkage, Comment, Declaration, SubHeading,
                           SuperClass);
}

ObjCMethodRecord *APISet::addObjCMethod(
    ObjCContainerRecord *Container, StringRef Name, StringRef USR,
    PresumedLoc Loc, const AvailabilityInfo &Availability,
    const DocComment &Comment, DeclarationFragments Declaration,
    DeclarationFragments SubHeading, FunctionSignature Signature,
    bool IsInstanceMethod) {
  auto Record = std::make_unique<ObjCMethodRecord>(
      Name, USR, Loc, Availability, Comment, Declaration, SubHeading, Signature,
      IsInstanceMethod);
  return Container->Methods.emplace_back(std::move(Record)).get();
}

ObjCPropertyRecord *APISet::addObjCProperty(
    ObjCContainerRecord *Container, StringRef Name, StringRef USR,
    PresumedLoc Loc, const AvailabilityInfo &Availability,
    const DocComment &Comment, DeclarationFragments Declaration,
    DeclarationFragments SubHeading,
    ObjCPropertyRecord::AttributeKind Attributes, StringRef GetterName,
    StringRef SetterName, bool IsOptional) {
  auto Record = std::make_unique<ObjCPropertyRecord>(
      Name, USR, Loc, Availability, Comment, Declaration, SubHeading,
      Attributes, GetterName, SetterName, IsOptional);
  return Container->Properties.emplace_back(std::move(Record)).get();
}

ObjCInstanceVariableRecord *APISet::addObjCInstanceVariable(
    ObjCContainerRecord *Container, StringRef Name, StringRef USR,
    PresumedLoc Loc, const AvailabilityInfo &Availability,
    const DocComment &Comment, DeclarationFragments Declaration,
    DeclarationFragments SubHeading,
    ObjCInstanceVariableRecord::AccessControl Access) {
  auto Record = std::make_unique<ObjCInstanceVariableRecord>(
      Name, USR, Loc, Availability, Comment, Declaration, SubHeading, Access);
  return Container->Ivars.emplace_back(std::move(Record)).get();
}

ObjCProtocolRecord *APISet::addObjCProtocol(
    StringRef Name, StringRef USR, PresumedLoc Loc,
    const AvailabilityInfo &Availability, const DocComment &Comment,
    DeclarationFragments Declaration, DeclarationFragments SubHeading) {
  return addTopLevelRecord(ObjCProtocols, Name, USR, Loc, Availability, Comment,
                           Declaration, SubHeading);
}

MacroDefinitionRecord *
APISet::addMacroDefinition(StringRef Name, StringRef USR, PresumedLoc Loc,
                           DeclarationFragments Declaration,
                           DeclarationFragments SubHeading) {
  return addTopLevelRecord(Macros, Name, USR, Loc, Declaration, SubHeading);
}

TypedefRecord *APISet::addTypedef(StringRef Name, StringRef USR,
                                  PresumedLoc Loc,
                                  const AvailabilityInfo &Availability,
                                  const DocComment &Comment,
                                  DeclarationFragments Declaration,
                                  DeclarationFragments SubHeading,
                                  SymbolReference UnderlyingType) {
  return addTopLevelRecord(Typedefs, Name, USR, Loc, Availability, Comment,
                           Declaration, SubHeading, UnderlyingType);
}

StringRef APISet::recordUSR(const Decl *D) {
  SmallString<128> USR;
  index::generateUSRForDecl(D, USR);
  return copyString(USR);
}

StringRef APISet::recordUSRForMacro(StringRef Name, SourceLocation SL,
                                    const SourceManager &SM) {
  SmallString<128> USR;
  index::generateUSRForMacro(Name, SL, SM, USR);
  return copyString(USR);
}

StringRef APISet::copyString(StringRef String) {
  if (String.empty())
    return {};

  // No need to allocate memory and copy if the string has already been stored.
  if (StringAllocator.identifyObject(String.data()))
    return String;

  void *Ptr = StringAllocator.Allocate(String.size(), 1);
  memcpy(Ptr, String.data(), String.size());
  return StringRef(reinterpret_cast<const char *>(Ptr), String.size());
}

APIRecord::~APIRecord() {}

ObjCContainerRecord::~ObjCContainerRecord() {}

void GlobalRecord::anchor() {}
void EnumConstantRecord::anchor() {}
void EnumRecord::anchor() {}
void StructFieldRecord::anchor() {}
void StructRecord::anchor() {}
void ObjCPropertyRecord::anchor() {}
void ObjCInstanceVariableRecord::anchor() {}
void ObjCMethodRecord::anchor() {}
void ObjCInterfaceRecord::anchor() {}
void ObjCProtocolRecord::anchor() {}
void MacroDefinitionRecord::anchor() {}
void TypedefRecord::anchor() {}
