//=- ClangBuiltinsEmitter.cpp - Generate Clang builtin templates-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits Clang's builtin templates.
//
//===----------------------------------------------------------------------===//

#include "TableGenBackends.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <sstream>

using namespace llvm;

static std::string TemplateNameList;
static std::string CreateBuiltinTemplateParameterList;

static llvm::StringSet<> BuiltinClasses;

namespace {
struct ParserState {
  size_t UniqueCounter = 0;
  size_t CurrentDepth = 0;
  bool EmittedSizeTInfo = false;
  bool EmittedUint32TInfo = false;
};

std::pair<std::string, std::string>
ParseTemplateParameterList(ParserState &PS,
                           ArrayRef<const Record *> TemplateArgs) {
  llvm::SmallVector<std::string, 4> Params;
  llvm::StringMap<std::string> TemplateNameToParmName;

  std::ostringstream Code;
  Code << std::boolalpha;

  size_t Position = 0;
  for (const Record *Arg : TemplateArgs) {
    std::string ParmName = "Parm" + std::to_string(PS.UniqueCounter++);
    if (Arg->isSubClassOf("Template")) {
      ++PS.CurrentDepth;
      auto [TemplateCode, TPLName] =
          ParseTemplateParameterList(PS, Arg->getValueAsListOfDefs("Args"));
      --PS.CurrentDepth;
      Code << TemplateCode << " auto *" << ParmName
           << " = TemplateTemplateParmDecl::Create(C, DC, SourceLocation(), "
           << PS.CurrentDepth << ", " << Position++
           << ", /*ParameterPack=*/false, /*Id=*/nullptr, /*Typename=*/false, "
           << TPLName << ");\n";
    } else if (Arg->isSubClassOf("Class")) {
      Code << " auto *" << ParmName
           << " = TemplateTypeParmDecl::Create(C, DC, SourceLocation(), "
              "SourceLocation(), "
           << PS.CurrentDepth << ", " << Position++
           << ", /*Id=*/nullptr, /*Typename=*/false, "
           << Arg->getValueAsBit("IsVariadic") << ");\n";
    } else if (Arg->isSubClassOf("NTTP")) {
      auto Type = Arg->getValueAsString("TypeName");

      if (!TemplateNameToParmName.contains(Type.str()))
        PrintFatalError("Unknown Type Name");

      auto TSIName = "TSI" + std::to_string(PS.UniqueCounter++);
      Code << " auto *" << TSIName << " = C.getTrivialTypeSourceInfo(QualType("
           << TemplateNameToParmName[Type.str()] << "->getTypeForDecl(), 0));\n"
           << " auto *" << ParmName
           << " = NonTypeTemplateParmDecl::Create(C, DC, SourceLocation(), "
              "SourceLocation(), "
           << PS.CurrentDepth << ", " << Position++ << ", /*Id=*/nullptr, "
           << TSIName << "->getType(), " << Arg->getValueAsBit("IsVariadic")
           << ", " << TSIName << ");\n";
    } else if (Arg->isSubClassOf("BuiltinNTTP")) {
      std::string SourceInfo;
      if (Arg->getValueAsString("TypeName") == "size_t") {
        SourceInfo = "SizeTInfo";
        if (!PS.EmittedSizeTInfo) {
          Code << "TypeSourceInfo *SizeTInfo = "
                  "C.getTrivialTypeSourceInfo(C.getSizeType());\n";
          PS.EmittedSizeTInfo = true;
        }
      } else if (Arg->getValueAsString("TypeName") == "uint32_t") {
        SourceInfo = "Uint32TInfo";
        if (!PS.EmittedUint32TInfo) {
          Code << "TypeSourceInfo *Uint32TInfo = "
                  "C.getTrivialTypeSourceInfo(C.UnsignedIntTy);\n";
          PS.EmittedUint32TInfo = true;
        }
      } else {
        PrintFatalError("Unknown Type Name");
      }
      Code << " auto *" << ParmName
           << " = NonTypeTemplateParmDecl::Create(C, DC, SourceLocation(), "
              "SourceLocation(), "
           << PS.CurrentDepth << ", " << Position++ << ", /*Id=*/nullptr, "
           << SourceInfo
           << "->getType(), "
              "/*ParameterPack=*/false, "
           << SourceInfo << ");\n";
    } else {
      PrintFatalError("Unknown Argument Type");
    }

    TemplateNameToParmName[Arg->getValueAsString("Name").str()] = ParmName;
    Params.emplace_back(std::move(ParmName));
  }

  auto TPLName = "TPL" + std::to_string(PS.UniqueCounter++);
  Code << " auto *" << TPLName
       << " = TemplateParameterList::Create(C, SourceLocation(), "
          "SourceLocation(), {";

  if (Params.empty()) {
    PrintFatalError(
        "Expected at least one argument in template parameter list");
  }

  bool First = true;
  for (const auto &e : Params) {
    if (First) {
      First = false;
      Code << e;
    } else {
      Code << ", " << e;
    }
  }
  Code << "}, SourceLocation(), nullptr);\n";

  return {std::move(Code).str(), std::move(TPLName)};
}

static void
EmitCreateBuiltinTemplateParameterList(std::vector<const Record *> TemplateArgs,
                                       StringRef Name) {
  using namespace std::string_literals;
  CreateBuiltinTemplateParameterList +=
      "case BTK"s + std::string{Name} + ": {\n"s;

  ParserState PS;
  auto [Code, TPLName] = ParseTemplateParameterList(PS, TemplateArgs);
  CreateBuiltinTemplateParameterList += Code + "\n  return " + TPLName + ";\n";

  CreateBuiltinTemplateParameterList += "  }\n";
}

void EmitBuiltinTemplate(const Record *BuiltinTemplate) {
  auto Class = BuiltinTemplate->getType()->getAsString();
  auto Name = BuiltinTemplate->getName();

  std::vector<const Record *> TemplateHead =
      BuiltinTemplate->getValueAsListOfDefs("TemplateHead");

  EmitCreateBuiltinTemplateParameterList(TemplateHead, Name);

  TemplateNameList += Class + "(";
  TemplateNameList += Name;
  TemplateNameList += ")\n";

  BuiltinClasses.insert(Class);
}

void EmitDefaultDefine(llvm::raw_ostream &OS, StringRef Name) {
  OS << "#ifndef " << Name << "\n";
  OS << "#define " << Name << "(NAME)" << " " << "BuiltinTemplate"
     << "(NAME)\n";
  OS << "#endif\n\n";
}

void EmitUndef(llvm::raw_ostream &OS, StringRef Name) {
  OS << "#undef " << Name << "\n";
}
} // namespace

void clang::EmitClangBuiltinTemplates(const llvm::RecordKeeper &Records,
                                      llvm::raw_ostream &OS) {
  emitSourceFileHeader("Tables and code for Clang's builtin templates", OS);

  for (const auto *Builtin :
       Records.getAllDerivedDefinitions("BuiltinTemplate"))
    EmitBuiltinTemplate(Builtin);

  for (const auto &ClassEntry : BuiltinClasses) {
    StringRef Class = ClassEntry.getKey();
    if (Class == "BuiltinTemplate")
      continue;
    EmitDefaultDefine(OS, Class);
  }

  OS << "#if defined(CREATE_BUILTIN_TEMPLATE_PARAMETER_LIST)\n"
     << CreateBuiltinTemplateParameterList
     << "#undef CREATE_BUILTIN_TEMPLATE_PARAMETER_LIST\n#else\n"
     << TemplateNameList << "#undef BuiltinTemplate\n#endif\n";

  for (const auto &ClassEntry : BuiltinClasses) {
    StringRef Class = ClassEntry.getKey();
    if (Class == "BuiltinTemplate")
      continue;
    EmitUndef(OS, Class);
  }
}
