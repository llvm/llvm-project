//===--------------- Implementation of PublicAPICommand ----------*-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PublicAPICommand.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

static const char NamedTypeClassName[] = "NamedType";
static const char PtrTypeClassName[] = "PtrType";
static const char RestrictedPtrTypeClassName[] = "RestrictedPtrType";
static const char ConstTypeClassName[] = "ConstType";
static const char StructTypeClassName[] = "Struct";

static const char StandardSpecClassName[] = "StandardSpec";
static const char PublicAPIClassName[] = "PublicAPI";

static bool isa(llvm::Record *Def, llvm::Record *TypeClass) {
  llvm::RecordRecTy *RecordType = Def->getType();
  llvm::ArrayRef<llvm::Record *> Classes = RecordType->getClasses();
  // We want exact types. That is, we don't want the classes listed in
  // spec.td to be subclassed. Hence, we do not want the record |Def|
  // to be of more than one class type..
  if (Classes.size() != 1)
    return false;
  return Classes[0] == TypeClass;
}

// Text blocks for macro definitions and type decls can be indented to
// suit the surrounding tablegen listing. We need to dedent such blocks
// before writing them out.
static void dedentAndWrite(llvm::StringRef Text, llvm::raw_ostream &OS) {
  llvm::SmallVector<llvm::StringRef, 10> Lines;
  llvm::SplitString(Text, Lines, "\n");
  size_t shortest_indent = 1024;
  for (llvm::StringRef L : Lines) {
    llvm::StringRef Indent = L.take_while([](char c) { return c == ' '; });
    size_t IndentSize = Indent.size();
    if (Indent.size() == L.size()) {
      // Line is all spaces so no point noting the indent.
      continue;
    }
    if (IndentSize < shortest_indent)
      shortest_indent = IndentSize;
  }
  for (llvm::StringRef L : Lines) {
    if (L.size() >= shortest_indent)
      OS << L.drop_front(shortest_indent) << '\n';
  }
}

class APIGenerator {
  llvm::StringRef StdHeader;

  // TableGen classes in spec.td.
  llvm::Record *NamedTypeClass;
  llvm::Record *PtrTypeClass;
  llvm::Record *RestrictedPtrTypeClass;
  llvm::Record *ConstTypeClass;
  llvm::Record *StructClass;
  llvm::Record *StandardSpecClass;
  llvm::Record *PublicAPIClass;

  using NameToRecordMapping = std::unordered_map<std::string, llvm::Record *>;
  using NameSet = std::unordered_set<std::string>;

  // Mapping from names to records defining them.
  NameToRecordMapping MacroSpecMap;
  NameToRecordMapping TypeSpecMap;
  NameToRecordMapping FunctionSpecMap;
  NameToRecordMapping MacroDefsMap;
  NameToRecordMapping TypeDeclsMap;

  NameSet Structs;
  NameSet Functions;

  bool isaNamedType(llvm::Record *Def) { return isa(Def, NamedTypeClass); }

  bool isaStructType(llvm::Record *Def) { return isa(Def, StructClass); }

  bool isaPtrType(llvm::Record *Def) { return isa(Def, PtrTypeClass); }

  bool isaConstType(llvm::Record *Def) { return isa(Def, ConstTypeClass); }

  bool isaRestrictedPtrType(llvm::Record *Def) {
    return isa(Def, RestrictedPtrTypeClass);
  }

  bool isaStandardSpec(llvm::Record *Def) {
    return isa(Def, StandardSpecClass);
  }

  bool isaPublicAPI(llvm::Record *Def) { return isa(Def, PublicAPIClass); }

  std::string getTypeAsString(llvm::Record *TypeRecord) {
    if (isaNamedType(TypeRecord) || isaStructType(TypeRecord)) {
      return TypeRecord->getValueAsString("Name");
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

  void indexStandardSpecDef(llvm::Record *StandardSpec) {
    auto HeaderSpecList = StandardSpec->getValueAsListOfDefs("Headers");
    for (llvm::Record *HeaderSpec : HeaderSpecList) {
      if (HeaderSpec->getValueAsString("Name") == StdHeader) {
        auto MacroSpecList = HeaderSpec->getValueAsListOfDefs("Macros");
        // TODO: Trigger a fatal error on duplicate specs.
        for (llvm::Record *MacroSpec : MacroSpecList)
          MacroSpecMap[MacroSpec->getValueAsString("Name")] = MacroSpec;

        auto TypeSpecList = HeaderSpec->getValueAsListOfDefs("Types");
        for (llvm::Record *TypeSpec : TypeSpecList)
          TypeSpecMap[TypeSpec->getValueAsString("Name")] = TypeSpec;

        auto FunctionSpecList = HeaderSpec->getValueAsListOfDefs("Functions");
        for (llvm::Record *FunctionSpec : FunctionSpecList) {
          FunctionSpecMap[FunctionSpec->getValueAsString("Name")] =
              FunctionSpec;
        }
      }
    }
  }

  void indexPublicAPIDef(llvm::Record *PublicAPI) {
    // While indexing the public API, we do not check if any of the entities
    // requested is from an included standard. Such a check is done while
    // generating the API.
    auto MacroDefList = PublicAPI->getValueAsListOfDefs("Macros");
    for (llvm::Record *MacroDef : MacroDefList)
      MacroDefsMap[MacroDef->getValueAsString("Name")] = MacroDef;

    auto TypeDeclList = PublicAPI->getValueAsListOfDefs("TypeDeclarations");
    for (llvm::Record *TypeDecl : TypeDeclList)
      TypeDeclsMap[TypeDecl->getValueAsString("Name")] = TypeDecl;

    auto StructList = PublicAPI->getValueAsListOfStrings("Structs");
    for (llvm::StringRef StructName : StructList)
      Structs.insert(StructName);

    auto FunctionList = PublicAPI->getValueAsListOfStrings("Functions");
    for (llvm::StringRef FunctionName : FunctionList)
      Functions.insert(FunctionName);
  }

  void index(llvm::RecordKeeper &Records) {
    NamedTypeClass = Records.getClass(NamedTypeClassName);
    PtrTypeClass = Records.getClass(PtrTypeClassName);
    RestrictedPtrTypeClass = Records.getClass(RestrictedPtrTypeClassName);
    StructClass = Records.getClass(StructTypeClassName);
    ConstTypeClass = Records.getClass(ConstTypeClassName);
    StandardSpecClass = Records.getClass(StandardSpecClassName);
    PublicAPIClass = Records.getClass(PublicAPIClassName);

    const auto &DefsMap = Records.getDefs();
    for (auto &Pair : DefsMap) {
      llvm::Record *Def = Pair.second.get();
      if (isaStandardSpec(Def))
        indexStandardSpecDef(Def);
      if (isaPublicAPI(Def)) {
        if (Def->getValueAsString("HeaderName") == StdHeader)
          indexPublicAPIDef(Def);
      }
    }
  }

public:
  APIGenerator(llvm::StringRef Header, llvm::RecordKeeper &Records)
      : StdHeader(Header) {
    index(Records);
  }

  void write(llvm::raw_ostream &OS) {
    for (auto &Pair : MacroDefsMap) {
      const std::string &Name = Pair.first;
      if (MacroSpecMap.find(Name) == MacroSpecMap.end())
        llvm::PrintFatalError(Name + " not found in any standard spec.\n");

      llvm::Record *MacroDef = Pair.second;
      dedentAndWrite(MacroDef->getValueAsString("Defn"), OS);

      OS << '\n';
    }

    for (auto &Pair : TypeDeclsMap) {
      const std::string &Name = Pair.first;
      if (TypeSpecMap.find(Name) == TypeSpecMap.end())
        llvm::PrintFatalError(Name + " not found in any standard spec.\n");

      llvm::Record *TypeDecl = Pair.second;
      dedentAndWrite(TypeDecl->getValueAsString("Decl"), OS);

      OS << '\n';
    }

    OS << "__BEGIN_C_DECLS\n\n";
    for (auto &Name : Functions) {
      if (FunctionSpecMap.find(Name) == FunctionSpecMap.end())
        llvm::PrintFatalError(Name + " not found in any standard spec.\n");

      llvm::Record *FunctionSpec = FunctionSpecMap[Name];
      llvm::Record *RetValSpec = FunctionSpec->getValueAsDef("Return");
      llvm::Record *ReturnType = RetValSpec->getValueAsDef("ReturnType");

      OS << getTypeAsString(ReturnType) << " " << Name << "(";

      auto ArgsList = FunctionSpec->getValueAsListOfDefs("Args");
      for (size_t i = 0; i < ArgsList.size(); ++i) {
        llvm::Record *ArgType = ArgsList[i]->getValueAsDef("ArgType");
        OS << getTypeAsString(ArgType);
        if (i < ArgsList.size() - 1)
          OS << ", ";
      }

      OS << ");\n\n";
    }
    OS << "__END_C_DECLS\n";
  }
};

namespace llvm_libc {

void writePublicAPI(llvm::raw_ostream &OS, llvm::RecordKeeper &Records) {}

const char PublicAPICommand::Name[] = "public_api";

void PublicAPICommand::run(llvm::raw_ostream &OS, const ArgVector &Args,
                           llvm::StringRef StdHeader,
                           llvm::RecordKeeper &Records,
                           const Command::ErrorReporter &Reporter) const {
  if (Args.size() != 0) {
    Reporter.printFatalError("public_api command does not take any arguments.");
  }

  APIGenerator G(StdHeader, Records);
  G.write(OS);
}

} // namespace llvm_libc
