//===--- AST.cpp - Utility AST functions  -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/SourceManager.h"

namespace clang {
namespace clangd {
using namespace llvm;

SourceLocation findNameLoc(const clang::Decl* D) {
  const auto& SM = D->getASTContext().getSourceManager();
  // FIXME: Revisit the strategy, the heuristic is limitted when handling
  // macros, we should use the location where the whole definition occurs.
  SourceLocation SpellingLoc = SM.getSpellingLoc(D->getLocation());
  if (D->getLocation().isMacroID()) {
    std::string PrintLoc = SpellingLoc.printToString(SM);
    if (llvm::StringRef(PrintLoc).startswith("<scratch") ||
        llvm::StringRef(PrintLoc).startswith("<command line>")) {
      // We use the expansion location for the following symbols, as spelling
      // locations of these symbols are not interesting to us:
      //   * symbols formed via macro concatenation, the spelling location will
      //     be "<scratch space>"
      //   * symbols controlled and defined by a compile command-line option
      //     `-DName=foo`, the spelling location will be "<command line>".
      SpellingLoc = SM.getExpansionRange(D->getLocation()).first;
    }
  }
  return SpellingLoc;
}

} // namespace clangd
} // namespace clang
