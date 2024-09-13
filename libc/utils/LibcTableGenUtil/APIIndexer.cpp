//===-- Implementation of APIIndexer class --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "APIIndexer.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

namespace llvm_libc {

static const char NamedTypeClassName[] = "NamedType";
static const char PtrTypeClassName[] = "PtrType";
static const char RestrictedPtrTypeClassName[] = "RestrictedPtrType";
static const char ConstTypeClassName[] = "ConstType";
static const char StructTypeClassName[] = "Struct";

static const char StandardSpecClassName[] = "StandardSpec";
static const char PublicAPIClassName[] = "PublicAPI";

static bool isa(const llvm::Record *Def, const llvm::Record *TypeClass) {
  llvm::RecordRecTy *RecordType = Def->getType();
  llvm::ArrayRef<llvm::Record *> Classes = RecordType->getClasses();
  // We want exact types. That is, we don't want the classes listed in
  // spec.td to be subclassed. Hence, we do not want the record |Def|
  // to be of more than one class type..
  if (Classes.size() != 1)
    return false;
  return Classes[0] == TypeClass;
}

bool APIIndexer::isaNamedType(const llvm::Record *Def) {
  return isa(Def, NamedTypeClass);
}

bool APIIndexer::isaStructType(const llvm::Record *Def) {
  return isa(Def, StructClass);
}

bool APIIndexer::isaPtrType(const llvm::Record *Def) {
  return isa(Def, PtrTypeClass);
}

bool APIIndexer::isaConstType(const llvm::Record *Def) {
  return isa(Def, ConstTypeClass);
}

bool APIIndexer::isaRestrictedPtrType(const llvm::Record *Def) {
  return isa(Def, RestrictedPtrTypeClass);
}

bool APIIndexer::isaStandardSpec(const llvm::Record *Def) {
  return isa(Def, StandardSpecClass);
}

bool APIIndexer::isaPublicAPI(const llvm::Record *Def) {
  return isa(Def, PublicAPIClass);
}

std::string APIIndexer::getTypeAsString(const llvm::Record *TypeRecord) {
  if (isaNamedType(TypeRecord) || isaStructType(TypeRecord)) {
    return std::string(TypeRecord->getValueAsString("Name"));
  } else if (isaPtrType(TypeRecord)) {
    return getTypeAsString(TypeRecord->getValueAsDef("PointeeType")) + " *";
  } else if (isaConstType(TypeRecord)) {
    return std::string("const ") +
           getTypeAsString(TypeRecord->getValueAsDef("UnqualifiedType"));
  } else if (isaRestrictedPtrType(TypeRecord)) {
    return getTypeAsString(TypeRecord->getValueAsDef("PointeeType")) +
           " *__restrict";
  } else {
    llvm::PrintFatalError(TypeRecord->getLoc(), "Invalid type.\n");
  }
}

void APIIndexer::indexStandardSpecDef(const llvm::Record *StandardSpec) {
  auto HeaderSpecList = StandardSpec->getValueAsListOfDefs("Headers");
  for (llvm::Record *HeaderSpec : HeaderSpecList) {
    llvm::StringRef Header = HeaderSpec->getValueAsString("Name");
    if (!StdHeader.has_value() || Header == StdHeader) {
      PublicHeaders.emplace(Header);
      auto MacroSpecList = HeaderSpec->getValueAsListOfDefs("Macros");
      // TODO: Trigger a fatal error on duplicate specs.
      for (llvm::Record *MacroSpec : MacroSpecList)
        MacroSpecMap[std::string(MacroSpec->getValueAsString("Name"))] =
            MacroSpec;

      auto TypeSpecList = HeaderSpec->getValueAsListOfDefs("Types");
      for (llvm::Record *TypeSpec : TypeSpecList)
        TypeSpecMap[std::string(TypeSpec->getValueAsString("Name"))] = TypeSpec;

      auto FunctionSpecList = HeaderSpec->getValueAsListOfDefs("Functions");
      for (llvm::Record *FunctionSpec : FunctionSpecList) {
        auto FunctionName = std::string(FunctionSpec->getValueAsString("Name"));
        FunctionSpecMap[FunctionName] = FunctionSpec;
        FunctionToHeaderMap[FunctionName] = std::string(Header);
      }

      auto EnumerationSpecList =
          HeaderSpec->getValueAsListOfDefs("Enumerations");
      for (llvm::Record *EnumerationSpec : EnumerationSpecList) {
        EnumerationSpecMap[std::string(
            EnumerationSpec->getValueAsString("Name"))] = EnumerationSpec;
      }

      auto ObjectSpecList = HeaderSpec->getValueAsListOfDefs("Objects");
      for (llvm::Record *ObjectSpec : ObjectSpecList) {
        auto ObjectName = std::string(ObjectSpec->getValueAsString("Name"));
        ObjectSpecMap[ObjectName] = ObjectSpec;
        ObjectToHeaderMap[ObjectName] = std::string(Header);
      }
    }
  }
}

void APIIndexer::indexPublicAPIDef(const llvm::Record *PublicAPI) {
  // While indexing the public API, we do not check if any of the entities
  // requested is from an included standard. Such a check is done while
  // generating the API.
  auto MacroDefList = PublicAPI->getValueAsListOfDefs("Macros");
  for (llvm::Record *MacroDef : MacroDefList)
    MacroDefsMap[std::string(MacroDef->getValueAsString("Name"))] = MacroDef;

  auto TypeList = PublicAPI->getValueAsListOfStrings("Types");
  for (llvm::StringRef TypeName : TypeList)
    RequiredTypes.insert(std::string(TypeName));

  auto StructList = PublicAPI->getValueAsListOfStrings("Structs");
  for (llvm::StringRef StructName : StructList)
    Structs.insert(std::string(StructName));

  auto FunctionList = PublicAPI->getValueAsListOfStrings("Functions");
  for (llvm::StringRef FunctionName : FunctionList)
    Functions.insert(std::string(FunctionName));

  auto EnumerationList = PublicAPI->getValueAsListOfStrings("Enumerations");
  for (llvm::StringRef EnumerationName : EnumerationList)
    Enumerations.insert(std::string(EnumerationName));

  auto ObjectList = PublicAPI->getValueAsListOfStrings("Objects");
  for (llvm::StringRef ObjectName : ObjectList)
    Objects.insert(std::string(ObjectName));
}

void APIIndexer::index(const llvm::RecordKeeper &Records) {
  NamedTypeClass = Records.getClass(NamedTypeClassName);
  PtrTypeClass = Records.getClass(PtrTypeClassName);
  RestrictedPtrTypeClass = Records.getClass(RestrictedPtrTypeClassName);
  StructClass = Records.getClass(StructTypeClassName);
  ConstTypeClass = Records.getClass(ConstTypeClassName);
  StandardSpecClass = Records.getClass(StandardSpecClassName);
  PublicAPIClass = Records.getClass(PublicAPIClassName);

  const auto &DefsMap = Records.getDefs();
  for (auto &Pair : DefsMap) {
    const llvm::Record *Def = Pair.second.get();
    if (isaStandardSpec(Def))
      indexStandardSpecDef(Def);
    if (isaPublicAPI(Def)) {
      if (!StdHeader.has_value() ||
          Def->getValueAsString("HeaderName") == StdHeader)
        indexPublicAPIDef(Def);
    }
  }
}

} // namespace llvm_libc
