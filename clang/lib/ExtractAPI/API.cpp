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
#include <memory>

using namespace clang::extractapi;
using namespace llvm;

GlobalRecord *APISet::addGlobal(GVKind Kind, StringRef Name, StringRef USR,
                                PresumedLoc Loc,
                                const AvailabilityInfo &Availability,
                                LinkageInfo Linkage, const DocComment &Comment,
                                DeclarationFragments Fragments,
                                DeclarationFragments SubHeading,
                                FunctionSignature Signature) {
  auto Result = Globals.insert({Name, nullptr});
  if (Result.second) {
    // Create the record if it does not already exist.
    auto Record = std::make_unique<GlobalRecord>(
        Kind, Name, USR, Loc, Availability, Linkage, Comment, Fragments,
        SubHeading, Signature);
    Result.first->second = std::move(Record);
  }
  return Result.first->second.get();
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
  auto Result = Enums.insert({Name, nullptr});
  if (Result.second) {
    // Create the record if it does not already exist.
    auto Record = std::make_unique<EnumRecord>(
        Name, USR, Loc, Availability, Comment, Declaration, SubHeading);
    Result.first->second = std::move(Record);
  }
  return Result.first->second.get();
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
  auto Result = Structs.insert({Name, nullptr});
  if (Result.second) {
    // Create the record if it does not already exist.
    auto Record = std::make_unique<StructRecord>(
        Name, USR, Loc, Availability, Comment, Declaration, SubHeading);
    Result.first->second = std::move(Record);
  }
  return Result.first->second.get();
}

ObjCInterfaceRecord *APISet::addObjCInterface(
    StringRef Name, StringRef USR, PresumedLoc Loc,
    const AvailabilityInfo &Availability, LinkageInfo Linkage,
    const DocComment &Comment, DeclarationFragments Declaration,
    DeclarationFragments SubHeading, SymbolReference SuperClass) {
  auto Result = ObjCInterfaces.insert({Name, nullptr});
  if (Result.second) {
    // Create the record if it does not already exist.
    auto Record = std::make_unique<ObjCInterfaceRecord>(
        Name, USR, Loc, Availability, Linkage, Comment, Declaration, SubHeading,
        SuperClass);
    Result.first->second = std::move(Record);
  }
  return Result.first->second.get();
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

StringRef APISet::recordUSR(const Decl *D) {
  SmallString<128> USR;
  index::generateUSRForDecl(D, USR);
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
