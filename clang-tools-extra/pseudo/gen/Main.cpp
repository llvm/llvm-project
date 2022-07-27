//===--- Main.cpp - Compile BNF grammar -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a tool to compile a BNF grammar, it is used by the build system to
// generate a necessary data bits to statically construct core pieces (Grammar,
// LRTable etc) of the LR parser.
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/grammar/Grammar.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include <algorithm>

using llvm::cl::desc;
using llvm::cl::init;
using llvm::cl::opt;
using llvm::cl::Required;
using llvm::cl::value_desc;
using llvm::cl::values;

namespace {
enum EmitType {
  EmitSymbolList,
  EmitGrammarContent,
};

opt<std::string> Grammar("grammar", desc("Parse a BNF grammar file."),
                         Required);
opt<EmitType>
    Emit(desc("which information to emit:"),
         values(clEnumValN(EmitSymbolList, "emit-symbol-list",
                           "Print nonterminal symbols (default)"),
                clEnumValN(EmitGrammarContent, "emit-grammar-content",
                           "Print the BNF grammar content as a string")));

opt<std::string> OutputFilename("o", init("-"), desc("Output"),
                                value_desc("file"));

std::string readOrDie(llvm::StringRef Path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
      llvm::MemoryBuffer::getFile(Path);
  if (std::error_code EC = Text.getError()) {
    llvm::errs() << "Error: can't read grammar file '" << Path
                 << "': " << EC.message() << "\n";
    ::exit(1);
  }
  return Text.get()->getBuffer().str();
}
} // namespace

namespace clang {
namespace pseudo {
namespace {

// Mangles a symbol name into a valid identifier.
//
// These follow names in the grammar fairly closely:
//   nonterminal: `ptr-declartor` becomes `ptr_declarator`;
//   punctuator: `,` becomes `COMMA`;
//   keyword: `INT` becomes `INT`;
//   terminal: `IDENTIFIER` becomes `IDENTIFIER`;
std::string mangleSymbol(SymbolID SID, const Grammar &G) {
  static std::string *TokNames = new std::string[]{
#define TOK(X) llvm::StringRef(#X).upper(),
#define KEYWORD(Keyword, Condition) llvm::StringRef(#Keyword).upper(),
#include "clang/Basic/TokenKinds.def"
      };
  if (isToken(SID))
    return TokNames[symbolToToken(SID)];
  std::string Name = G.symbolName(SID).str();
  // translation-unit -> translation_unit
  std::replace(Name.begin(), Name.end(), '-', '_');
  return Name;
}

// Mangles the RHS of a rule definition into a valid identifier.
// 
// These are unique only for a fixed LHS.
// e.g. for the grammar rule `ptr-declarator := ptr-operator ptr-declarator`,
// it is `ptr_operator__ptr_declarator`.
std::string mangleRule(RuleID RID, const Grammar &G) {
  const auto &R = G.lookupRule(RID);
  std::string MangleName = mangleSymbol(R.seq().front(), G);
  for (SymbolID S : R.seq().drop_front()) {
    MangleName.append("__");
    MangleName.append(mangleSymbol(S, G));
  }
  return MangleName;
}

} // namespace
} // namespace pseudo
} // namespace clang

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "");

  std::string GrammarText = readOrDie(Grammar);
  std::vector<std::string> Diags;
  auto G = clang::pseudo::Grammar::parseBNF(GrammarText, Diags);

  if (!Diags.empty()) {
    llvm::errs() << llvm::join(Diags, "\n");
    return 1;
  }

  std::error_code EC;
  llvm::ToolOutputFile Out{OutputFilename, EC, llvm::sys::fs::OF_None};
  if (EC) {
    llvm::errs() << EC.message() << '\n';
    return 1;
  }

  switch (Emit) {
  case EmitSymbolList:
    Out.os() << R"cpp(
#ifndef NONTERMINAL
#define NONTERMINAL(NAME, ID)
#endif
#ifndef RULE
#define RULE(LHS, RHS, ID)
#endif
#ifndef EXTENSION
#define EXTENSION(NAME, ID)
#endif
)cpp";
    for (clang::pseudo::SymbolID ID = 0; ID < G.table().Nonterminals.size();
         ++ID) {
      Out.os() << llvm::formatv("NONTERMINAL({0}, {1})\n",
                                clang::pseudo::mangleSymbol(ID, G), ID);
      for (const clang::pseudo::Rule &R : G.rulesFor(ID)) {
        clang::pseudo::RuleID RID = &R - G.table().Rules.data();
        Out.os() << llvm::formatv("RULE({0}, {1}, {2})\n",
                                  clang::pseudo::mangleSymbol(R.Target, G),
                                  clang::pseudo::mangleRule(RID, G), RID);
      }
    }
    for (clang::pseudo::ExtensionID EID = 1 /*skip the sentinel 0 value*/;
         EID < G.table().AttributeValues.size(); ++EID) {
      llvm::StringRef Name = G.table().AttributeValues[EID];
      assert(!Name.empty());
      Out.os() << llvm::formatv("EXTENSION({0}, {1})\n", Name, EID);
    }
    Out.os() << R"cpp(
#undef NONTERMINAL
#undef RULE
#undef EXTENSION
)cpp";
    break;
  case EmitGrammarContent:
    for (llvm::StringRef Line : llvm::split(GrammarText, '\n')) {
      Out.os() << '"';
      Out.os().write_escaped((Line + "\n").str());
      Out.os() << "\"\n";
    }
    break;
  }

  Out.keep();

  return 0;
}
