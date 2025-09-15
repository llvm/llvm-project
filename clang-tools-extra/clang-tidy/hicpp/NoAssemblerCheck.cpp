//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoAssemblerCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::hicpp {

namespace {
AST_MATCHER(VarDecl, isAsm) { return Node.hasAttr<clang::AsmLabelAttr>(); }
const ast_matchers::internal::VariadicDynCastAllOfMatcher<Decl,
                                                          FileScopeAsmDecl>
    fileScopeAsmDecl; // NOLINT(readability-identifier-*) preserve clang style
} // namespace

void NoAssemblerCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(asmStmt().bind("asm-stmt"), this);
  Finder->addMatcher(fileScopeAsmDecl().bind("asm-file-scope"), this);
  Finder->addMatcher(varDecl(isAsm()).bind("asm-var"), this);
}

void NoAssemblerCheck::check(const MatchFinder::MatchResult &Result) {
  SourceLocation ASMLocation;
  if (const auto *ASM = Result.Nodes.getNodeAs<AsmStmt>("asm-stmt"))
    ASMLocation = ASM->getAsmLoc();
  else if (const auto *ASM =
               Result.Nodes.getNodeAs<FileScopeAsmDecl>("asm-file-scope"))
    ASMLocation = ASM->getAsmLoc();
  else if (const auto *ASM = Result.Nodes.getNodeAs<VarDecl>("asm-var"))
    ASMLocation = ASM->getLocation();
  else
    llvm_unreachable("Unhandled case in matcher.");

  diag(ASMLocation, "do not use inline assembler in safety-critical code");
}

} // namespace clang::tidy::hicpp
