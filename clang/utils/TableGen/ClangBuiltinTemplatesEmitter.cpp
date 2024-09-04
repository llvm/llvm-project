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

using namespace llvm;

std::string TemplateNameList;
std::string CreateBuiltinTemplateParameterList;

namespace {
struct ParserState {
  size_t UniqueCounter = 0;
  size_t CurrentDepth = 0;
  bool EmittedSizeTInfo = false;
};

std::pair<std::string, std::string>
ParseTemplateParameterList(ParserState &PS, StringRef &TemplateParmList) {
  auto Alphabetic = [](char c) { return std::isalpha(c); };
  auto BoolToStr = [](bool b) { return b ? "true" : "false"; };

  std::string Generator;
  std::vector<std::string> Params;
  std::unordered_map<std::string, std::string> TemplateNameToParmName;
  TemplateParmList = TemplateParmList.ltrim();
  if (!TemplateParmList.consume_front("<"))
    PrintFatalError("Expected '<' to start the parameter list");

  size_t Position = 0;
  while (true) {
    std::string ParmName = "Parm" + std::to_string(PS.UniqueCounter++);
    if (TemplateParmList.consume_front("size_t")) {
      if (!PS.EmittedSizeTInfo) {
        PS.EmittedSizeTInfo = true;
        Generator += R"C++(
  TypeSourceInfo *SizeTInfo = C.getTrivialTypeSourceInfo(C.getSizeType());
)C++";
      }

      Generator += "  auto *" + ParmName + R"C++(
    = NonTypeTemplateParmDecl::Create(C, DC, SourceLocation(), SourceLocation(),
        )C++" + std::to_string(PS.CurrentDepth) +
                   ", " + std::to_string(Position++) + R"C++(, /*Id=*/nullptr,
        SizeTInfo->getType(), /*ParameterPack=*/false, SizeTInfo);
)C++";
    } else if (TemplateParmList.consume_front("class")) {
      bool ParameterPack = TemplateParmList.consume_front("...");

      Generator += "  auto *" + ParmName + R"C++(
    = TemplateTypeParmDecl::Create(C, DC, SourceLocation(), SourceLocation(),
      )C++" + std::to_string(PS.CurrentDepth) +
                   ", " + std::to_string(Position++) +
                   R"C++(, /*Id=*/nullptr, /*Typename=*/false, )C++" +
                   BoolToStr(ParameterPack) + ");\n";
    } else if (TemplateParmList.consume_front("template")) {
      ++PS.CurrentDepth;
      auto [Code, TPLName] = ParseTemplateParameterList(PS, TemplateParmList);
      --PS.CurrentDepth;
      TemplateParmList = TemplateParmList.ltrim();
      if (!TemplateParmList.consume_front("class")) {
        PrintFatalError("Expected 'class' after template template list");
      }
      Generator += Code;
      Generator +=
          "  auto *" + ParmName + R"C++(
    = TemplateTemplateParmDecl::Create(C, DC, SourceLocation(), )C++" +
          std::to_string(PS.CurrentDepth) + ", " + std::to_string(Position++) +
          ", /*ParameterPack=*/false, /*Id=*/nullptr, /*Typename=*/false, " +
          TPLName + ");\n";
    } else {
      auto Name = TemplateParmList.take_while(Alphabetic).str();
      if (TemplateNameToParmName.find(Name) != TemplateNameToParmName.end()) {
        TemplateParmList = TemplateParmList.drop_front(Name.size());
        bool ParameterPack = TemplateParmList.consume_front("...");

        auto TSIName = "TSI" + std::to_string(PS.UniqueCounter++);
        Generator += "  auto *" + TSIName + R"C++(
    = C.getTrivialTypeSourceInfo(QualType()C++" +
                     TemplateNameToParmName[Name] +
                     R"C++(->getTypeForDecl(), 0));
  auto *)C++" + ParmName +
                     R"C++( = NonTypeTemplateParmDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), )C++" +
                     std::to_string(PS.CurrentDepth) + ", " +
                     std::to_string(Position++) + ", /*Id=*/nullptr, " +
                     TSIName + "->getType(), " + BoolToStr(ParameterPack) +
                     ", " + TSIName + ");\n";
      } else {
        PrintFatalError("Unknown argument");
      }
    }
    TemplateParmList = TemplateParmList.ltrim();
    auto ID = TemplateParmList.take_while(Alphabetic);
    if (!ID.empty()) {
      TemplateNameToParmName[ID.str()] = ParmName;
      TemplateParmList = TemplateParmList.drop_front(ID.size());
    }

    Params.emplace_back(std::move(ParmName));

    if (!TemplateParmList.consume_front(","))
      break;
    TemplateParmList = TemplateParmList.ltrim();
  }

  if (!TemplateParmList.consume_front(">")) {
    PrintWarning("Expected '>' to close template parameter list");
    PrintWarning(TemplateParmList);
  }

  auto TPLName = "TPL" + std::to_string(PS.UniqueCounter++);
  Generator += "  auto *" + TPLName + R"C++( = TemplateParameterList::Create(
      C, SourceLocation(), SourceLocation(), {)C++";

  if (Params.empty())
    PrintFatalError(
        "Expected at least one argument in template parameter list");

  bool First = true;
  for (auto e : Params) {
    if (First) {
      First = false;
      Generator += e;
    } else {
      Generator += ", " + e;
    }
  }
  Generator += "}, SourceLocation(), nullptr);\n";

  return {std::move(Generator), std::move(TPLName)};
}

void EmitCreateBuiltinTemplateParameterList(StringRef Prototype,
                                            StringRef Name) {
  using namespace std::string_literals;
  CreateBuiltinTemplateParameterList +=
      "case BTK"s + std::string{Name} + ": {\n"s;
  if (!Prototype.consume_front("template"))
    PrintFatalError(
        "Expected template prototype to start with 'template' keyword");

  ParserState PS;
  auto [Code, TPLName] = ParseTemplateParameterList(PS, Prototype);
  CreateBuiltinTemplateParameterList += Code + "\n  return " + TPLName + ";\n";

  CreateBuiltinTemplateParameterList += "  }\n";
}

void EmitBuiltinTemplate(raw_ostream &OS, const Record *BuiltinTemplate) {
  auto Prototype = BuiltinTemplate->getValueAsString("Prototype");
  auto Name = BuiltinTemplate->getName();

  EmitCreateBuiltinTemplateParameterList(Prototype, Name);

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
