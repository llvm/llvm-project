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
RecordTy *addTopLevelRecord(DenseMap<StringRef, APIRecord *> &USRLookupTable,
                            APISet::RecordMap<RecordTy> &RecordMap,
                            StringRef USR, CtorArgsTy &&...CtorArgs) {
  auto Result = RecordMap.insert({USR, nullptr});

  // Create the record if it does not already exist
  if (Result.second)
    Result.first->second =
        std::make_unique<RecordTy>(USR, std::forward<CtorArgsTy>(CtorArgs)...);

  auto *Record = Result.first->second.get();
  USRLookupTable.insert({USR, Record});
  return Record;
}

} // namespace

GlobalVariableRecord *
APISet::addGlobalVar(StringRef Name, StringRef USR, PresumedLoc Loc,
                     AvailabilitySet Availabilities, LinkageInfo Linkage,
                     const DocComment &Comment, DeclarationFragments Fragments,
                     DeclarationFragments SubHeading, bool IsFromSystemHeader) {
  return addTopLevelRecord(USRBasedLookupTable, GlobalVariables, USR, Name, Loc,
                           std::move(Availabilities), Linkage, Comment,
                           Fragments, SubHeading, IsFromSystemHeader);
}

GlobalFunctionRecord *APISet::addGlobalFunction(
    StringRef Name, StringRef USR, PresumedLoc Loc,
    AvailabilitySet Availabilities, LinkageInfo Linkage,
    const DocComment &Comment, DeclarationFragments Fragments,
    DeclarationFragments SubHeading, FunctionSignature Signature,
    bool IsFromSystemHeader) {
  return addTopLevelRecord(USRBasedLookupTable, GlobalFunctions, USR, Name, Loc,
                           std::move(Availabilities), Linkage, Comment,
                           Fragments, SubHeading, Signature,
                           IsFromSystemHeader);
}

EnumConstantRecord *APISet::addEnumConstant(EnumRecord *Enum, StringRef Name,
                                            StringRef USR, PresumedLoc Loc,
                                            AvailabilitySet Availabilities,
                                            const DocComment &Comment,
                                            DeclarationFragments Declaration,
                                            DeclarationFragments SubHeading,
                                            bool IsFromSystemHeader) {
  auto Record = std::make_unique<EnumConstantRecord>(
      USR, Name, Loc, std::move(Availabilities), Comment, Declaration,
      SubHeading, IsFromSystemHeader);
  Record->ParentInformation = APIRecord::HierarchyInformation(
      Enum->USR, Enum->Name, Enum->getKind(), Enum);
  USRBasedLookupTable.insert({USR, Record.get()});
  return Enum->Constants.emplace_back(std::move(Record)).get();
}

EnumRecord *APISet::addEnum(StringRef Name, StringRef USR, PresumedLoc Loc,
                            AvailabilitySet Availabilities,
                            const DocComment &Comment,
                            DeclarationFragments Declaration,
                            DeclarationFragments SubHeading,
                            bool IsFromSystemHeader) {
  return addTopLevelRecord(USRBasedLookupTable, Enums, USR, Name, Loc,
                           std::move(Availabilities), Comment, Declaration,
                           SubHeading, IsFromSystemHeader);
}

StructFieldRecord *APISet::addStructField(StructRecord *Struct, StringRef Name,
                                          StringRef USR, PresumedLoc Loc,
                                          AvailabilitySet Availabilities,
                                          const DocComment &Comment,
                                          DeclarationFragments Declaration,
                                          DeclarationFragments SubHeading,
                                          bool IsFromSystemHeader) {
  auto Record = std::make_unique<StructFieldRecord>(
      USR, Name, Loc, std::move(Availabilities), Comment, Declaration,
      SubHeading, IsFromSystemHeader);
  Record->ParentInformation = APIRecord::HierarchyInformation(
      Struct->USR, Struct->Name, Struct->getKind(), Struct);
  USRBasedLookupTable.insert({USR, Record.get()});
  return Struct->Fields.emplace_back(std::move(Record)).get();
}

StructRecord *APISet::addStruct(StringRef Name, StringRef USR, PresumedLoc Loc,
                                AvailabilitySet Availabilities,
                                const DocComment &Comment,
                                DeclarationFragments Declaration,
                                DeclarationFragments SubHeading,
                                bool IsFromSystemHeader) {
  return addTopLevelRecord(USRBasedLookupTable, Structs, USR, Name, Loc,
                           std::move(Availabilities), Comment, Declaration,
                           SubHeading, IsFromSystemHeader);
}

ObjCCategoryRecord *APISet::addObjCCategory(
    StringRef Name, StringRef USR, PresumedLoc Loc,
    AvailabilitySet Availabilities, const DocComment &Comment,
    DeclarationFragments Declaration, DeclarationFragments SubHeading,
    SymbolReference Interface, bool IsFromSystemHeader) {
  // Create the category record.
  auto *Record =
      addTopLevelRecord(USRBasedLookupTable, ObjCCategories, USR, Name, Loc,
                        std::move(Availabilities), Comment, Declaration,
                        SubHeading, Interface, IsFromSystemHeader);

  // If this category is extending a known interface, associate it with the
  // ObjCInterfaceRecord.
  auto It = ObjCInterfaces.find(Interface.USR);
  if (It != ObjCInterfaces.end())
    It->second->Categories.push_back(Record);

  return Record;
}

ObjCInterfaceRecord *
APISet::addObjCInterface(StringRef Name, StringRef USR, PresumedLoc Loc,
                         AvailabilitySet Availabilities, LinkageInfo Linkage,
                         const DocComment &Comment,
                         DeclarationFragments Declaration,
                         DeclarationFragments SubHeading,
                         SymbolReference SuperClass, bool IsFromSystemHeader) {
  return addTopLevelRecord(USRBasedLookupTable, ObjCInterfaces, USR, Name, Loc,
                           std::move(Availabilities), Linkage, Comment,
                           Declaration, SubHeading, SuperClass,
                           IsFromSystemHeader);
}

ObjCMethodRecord *APISet::addObjCMethod(
    ObjCContainerRecord *Container, StringRef Name, StringRef USR,
    PresumedLoc Loc, AvailabilitySet Availabilities, const DocComment &Comment,
    DeclarationFragments Declaration, DeclarationFragments SubHeading,
    FunctionSignature Signature, bool IsInstanceMethod,
    bool IsFromSystemHeader) {
  std::unique_ptr<ObjCMethodRecord> Record;
  if (IsInstanceMethod)
    Record = std::make_unique<ObjCInstanceMethodRecord>(
        USR, Name, Loc, std::move(Availabilities), Comment, Declaration,
        SubHeading, Signature, IsFromSystemHeader);
  else
    Record = std::make_unique<ObjCClassMethodRecord>(
        USR, Name, Loc, std::move(Availabilities), Comment, Declaration,
        SubHeading, Signature, IsFromSystemHeader);

  Record->ParentInformation = APIRecord::HierarchyInformation(
      Container->USR, Container->Name, Container->getKind(), Container);
  USRBasedLookupTable.insert({USR, Record.get()});
  return Container->Methods.emplace_back(std::move(Record)).get();
}

ObjCPropertyRecord *APISet::addObjCProperty(
    ObjCContainerRecord *Container, StringRef Name, StringRef USR,
    PresumedLoc Loc, AvailabilitySet Availabilities, const DocComment &Comment,
    DeclarationFragments Declaration, DeclarationFragments SubHeading,
    ObjCPropertyRecord::AttributeKind Attributes, StringRef GetterName,
    StringRef SetterName, bool IsOptional, bool IsInstanceProperty,
    bool IsFromSystemHeader) {
  std::unique_ptr<ObjCPropertyRecord> Record;
  if (IsInstanceProperty)
    Record = std::make_unique<ObjCInstancePropertyRecord>(
        USR, Name, Loc, std::move(Availabilities), Comment, Declaration,
        SubHeading, Attributes, GetterName, SetterName, IsOptional,
        IsFromSystemHeader);
  else
    Record = std::make_unique<ObjCClassPropertyRecord>(
        USR, Name, Loc, std::move(Availabilities), Comment, Declaration,
        SubHeading, Attributes, GetterName, SetterName, IsOptional,
        IsFromSystemHeader);
  Record->ParentInformation = APIRecord::HierarchyInformation(
      Container->USR, Container->Name, Container->getKind(), Container);
  USRBasedLookupTable.insert({USR, Record.get()});
  return Container->Properties.emplace_back(std::move(Record)).get();
}

ObjCInstanceVariableRecord *APISet::addObjCInstanceVariable(
    ObjCContainerRecord *Container, StringRef Name, StringRef USR,
    PresumedLoc Loc, AvailabilitySet Availabilities, const DocComment &Comment,
    DeclarationFragments Declaration, DeclarationFragments SubHeading,
    ObjCInstanceVariableRecord::AccessControl Access, bool IsFromSystemHeader) {
  auto Record = std::make_unique<ObjCInstanceVariableRecord>(
      USR, Name, Loc, std::move(Availabilities), Comment, Declaration,
      SubHeading, Access, IsFromSystemHeader);
  Record->ParentInformation = APIRecord::HierarchyInformation(
      Container->USR, Container->Name, Container->getKind(), Container);
  USRBasedLookupTable.insert({USR, Record.get()});
  return Container->Ivars.emplace_back(std::move(Record)).get();
}

ObjCProtocolRecord *APISet::addObjCProtocol(StringRef Name, StringRef USR,
                                            PresumedLoc Loc,
                                            AvailabilitySet Availabilities,
                                            const DocComment &Comment,
                                            DeclarationFragments Declaration,
                                            DeclarationFragments SubHeading,
                                            bool IsFromSystemHeader) {
  return addTopLevelRecord(USRBasedLookupTable, ObjCProtocols, USR, Name, Loc,
                           std::move(Availabilities), Comment, Declaration,
                           SubHeading, IsFromSystemHeader);
}

MacroDefinitionRecord *
APISet::addMacroDefinition(StringRef Name, StringRef USR, PresumedLoc Loc,
                           DeclarationFragments Declaration,
                           DeclarationFragments SubHeading,
                           bool IsFromSystemHeader) {
  return addTopLevelRecord(USRBasedLookupTable, Macros, USR, Name, Loc,
                           Declaration, SubHeading, IsFromSystemHeader);
}

TypedefRecord *
APISet::addTypedef(StringRef Name, StringRef USR, PresumedLoc Loc,
                   AvailabilitySet Availabilities, const DocComment &Comment,
                   DeclarationFragments Declaration,
                   DeclarationFragments SubHeading,
                   SymbolReference UnderlyingType, bool IsFromSystemHeader) {
  return addTopLevelRecord(USRBasedLookupTable, Typedefs, USR, Name, Loc,
                           std::move(Availabilities), Comment, Declaration,
                           SubHeading, UnderlyingType, IsFromSystemHeader);
}

APIRecord *APISet::findRecordForUSR(StringRef USR) const {
  if (USR.empty())
    return nullptr;

  return USRBasedLookupTable.lookup(USR);
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
ObjCMethodRecord::~ObjCMethodRecord() {}
ObjCPropertyRecord::~ObjCPropertyRecord() {}

void GlobalFunctionRecord::anchor() {}
void GlobalVariableRecord::anchor() {}
void EnumConstantRecord::anchor() {}
void EnumRecord::anchor() {}
void StructFieldRecord::anchor() {}
void StructRecord::anchor() {}
void ObjCInstancePropertyRecord::anchor() {}
void ObjCClassPropertyRecord::anchor() {}
void ObjCInstanceVariableRecord::anchor() {}
void ObjCInstanceMethodRecord::anchor() {}
void ObjCClassMethodRecord::anchor() {}
void ObjCCategoryRecord::anchor() {}
void ObjCInterfaceRecord::anchor() {}
void ObjCProtocolRecord::anchor() {}
void MacroDefinitionRecord::anchor() {}
void TypedefRecord::anchor() {}
