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
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <sstream>

using namespace llvm;

static std::string TemplateNameList;
static std::string CreateBuiltinTemplateParameterList;

namespace {
struct ParserState {
  size_t UniqueCounter = 0;
  size_t CurrentDepth = 0;
  bool EmittedSizeTInfo = false;
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

      if (TemplateNameToParmName.find(Type.str()) ==
          TemplateNameToParmName.end()) {
        PrintFatalError("Unkown Type Name");
      }

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
      if (Arg->getValueAsString("TypeName") != "size_t")
        PrintFatalError("Unkown Type Name");
      if (!PS.EmittedSizeTInfo) {
        Code << "TypeSourceInfo *SizeTInfo = "
                "C.getTrivialTypeSourceInfo(C.getSizeType());\n";
        PS.EmittedSizeTInfo = true;
      }
      Code << " auto *" << ParmName
           << " = NonTypeTemplateParmDecl::Create(C, DC, SourceLocation(), "
              "SourceLocation(), "
           << PS.CurrentDepth << ", " << Position++
           << ", /*Id=*/nullptr, SizeTInfo->getType(), "
              "/*ParameterPack=*/false, SizeTInfo);\n";
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

void EmitBuiltinTemplate(raw_ostream &OS, const Record *BuiltinTemplate) {
  auto Name = BuiltinTemplate->getName();

  std::vector<const Record *> TemplateHead =
      BuiltinTemplate->getValueAsListOfDefs("TemplateHead");

  EmitCreateBuiltinTemplateParameterList(TemplateHead, Name);

  TemplateNameList += "BuiltinTemplate(";
  TemplateNameList += Name;
  TemplateNameList += ")\n";
}
} // namespace

void clang::EmitClangBuiltinTemplates(const llvm::RecordKeeper &Records,
                                      llvm::raw_ostream &OS) {
  emitSourceFileHeader("Tables and code for Clang's builtin templates", OS);
  for (const auto *Builtin :
       Records.getAllDerivedDefinitions("BuiltinTemplate"))
    EmitBuiltinTemplate(OS, Builtin);

  OS << "#if defined(CREATE_BUILTIN_TEMPLATE_PARAMETER_LIST)\n"
     << CreateBuiltinTemplateParameterList
     << "#undef CREATE_BUILTIN_TEMPLATE_PARAMETER_LIST\n#else\n"
     << TemplateNameList << "#undef BuiltinTemplate\n#endif\n";
}
