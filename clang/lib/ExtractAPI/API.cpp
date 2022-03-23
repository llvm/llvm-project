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
#include "llvm/Support/Allocator.h"

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
    auto Record = APIRecordUniquePtr<GlobalRecord>(new (Allocator) GlobalRecord{
        Kind, Name, USR, Loc, Availability, Linkage, Comment, Fragments,
        SubHeading, Signature});
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
  auto Record =
      APIRecordUniquePtr<EnumConstantRecord>(new (Allocator) EnumConstantRecord{
          Name, USR, Loc, Availability, Comment, Declaration, SubHeading});
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
    auto Record = APIRecordUniquePtr<EnumRecord>(new (Allocator) EnumRecord{
        Name, USR, Loc, Availability, Comment, Declaration, SubHeading});
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
  auto Record =
      APIRecordUniquePtr<StructFieldRecord>(new (Allocator) StructFieldRecord{
          Name, USR, Loc, Availability, Comment, Declaration, SubHeading});
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
    auto Record = APIRecordUniquePtr<StructRecord>(new (Allocator) StructRecord{
        Name, USR, Loc, Availability, Comment, Declaration, SubHeading});
    Result.first->second = std::move(Record);
  }
  return Result.first->second.get();
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
  if (Allocator.identifyObject(String.data()))
    return String;

  void *Ptr = Allocator.Allocate(String.size(), 1);
  memcpy(Ptr, String.data(), String.size());
  return StringRef(reinterpret_cast<const char *>(Ptr), String.size());
}

APIRecord::~APIRecord() {}

void GlobalRecord::anchor() {}
