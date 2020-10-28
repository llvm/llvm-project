//===-- Implementation of PublicAPICommand --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PublicAPICommand.h"

#include "utils/LibcTableGenUtil/APIIndexer.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Record.h"

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

namespace llvm_libc {

void writeAPIFromIndex(APIIndexer &G, llvm::raw_ostream &OS) {
  for (auto &Pair : G.MacroDefsMap) {
    const std::string &Name = Pair.first;
    if (G.MacroSpecMap.find(Name) == G.MacroSpecMap.end())
      llvm::PrintFatalError(Name + " not found in any standard spec.\n");

    llvm::Record *MacroDef = Pair.second;
    dedentAndWrite(MacroDef->getValueAsString("Defn"), OS);

    OS << '\n';
  }

  for (auto &Pair : G.TypeDeclsMap) {
    const std::string &Name = Pair.first;
    if (G.TypeSpecMap.find(Name) == G.TypeSpecMap.end())
      llvm::PrintFatalError(Name + " not found in any standard spec.\n");

    llvm::Record *TypeDecl = Pair.second;
    dedentAndWrite(TypeDecl->getValueAsString("Decl"), OS);

    OS << '\n';
  }

  if (G.Enumerations.size() != 0)
    OS << "enum {" << '\n';
  for (const auto &Name : G.Enumerations) {
    if (G.EnumerationSpecMap.find(Name) == G.EnumerationSpecMap.end())
      llvm::PrintFatalError(
          Name + " is not listed as an enumeration in any standard spec.\n");

    llvm::Record *EnumerationSpec = G.EnumerationSpecMap[Name];
    OS << "  " << EnumerationSpec->getValueAsString("Name");
    auto Value = EnumerationSpec->getValueAsString("Value");
    if (Value == "__default__") {
      OS << ",\n";
    } else {
      OS << " = " << Value << ",\n";
    }
  }
  if (G.Enumerations.size() != 0)
    OS << "};\n\n";

  OS << "__BEGIN_C_DECLS\n\n";
  for (auto &Name : G.Functions) {
    if (G.FunctionSpecMap.find(Name) == G.FunctionSpecMap.end())
      llvm::PrintFatalError(Name + " not found in any standard spec.\n");

    llvm::Record *FunctionSpec = G.FunctionSpecMap[Name];
    llvm::Record *RetValSpec = FunctionSpec->getValueAsDef("Return");
    llvm::Record *ReturnType = RetValSpec->getValueAsDef("ReturnType");

    OS << G.getTypeAsString(ReturnType) << " " << Name << "(";

    auto ArgsList = FunctionSpec->getValueAsListOfDefs("Args");
    for (size_t i = 0; i < ArgsList.size(); ++i) {
      llvm::Record *ArgType = ArgsList[i]->getValueAsDef("ArgType");
      OS << G.getTypeAsString(ArgType);
      if (i < ArgsList.size() - 1)
        OS << ", ";
    }

    OS << ");\n\n";
  }
  OS << "__END_C_DECLS\n";
}

void writePublicAPI(llvm::raw_ostream &OS, llvm::RecordKeeper &Records) {}

const char PublicAPICommand::Name[] = "public_api";

void PublicAPICommand::run(llvm::raw_ostream &OS, const ArgVector &Args,
                           llvm::StringRef StdHeader,
                           llvm::RecordKeeper &Records,
                           const Command::ErrorReporter &Reporter) const {
  if (Args.size() != 0) {
    Reporter.printFatalError("public_api command does not take any arguments.");
  }

  APIIndexer G(StdHeader, Records);
  writeAPIFromIndex(G, OS);
}

} // namespace llvm_libc
