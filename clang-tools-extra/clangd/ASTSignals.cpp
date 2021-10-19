//===--- ASTSignals.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTSignals.h"
#include "AST.h"
#include "FindTarget.h"

namespace clang {
namespace clangd {
ASTSignals ASTSignals::derive(const ParsedAST &AST) {
  ASTSignals Signals;
  const SourceManager &SM = AST.getSourceManager();
  findExplicitReferences(
      AST.getASTContext(),
      [&](ReferenceLoc Ref) {
        for (const NamedDecl *ND : Ref.Targets) {
          if (!isInsideMainFile(Ref.NameLoc, SM))
            continue;
          SymbolID ID = getSymbolID(ND);
          if (!ID)
            continue;
          unsigned &SymbolCount = Signals.ReferencedSymbols[ID];
          SymbolCount++;
          // Process namespace only when we see the symbol for the first time.
          if (SymbolCount != 1)
            continue;
          if (const auto *NSD = dyn_cast<NamespaceDecl>(ND->getDeclContext())) {
            if (NSD->isAnonymousNamespace())
              continue;
            std::string NS = printNamespaceScope(*NSD);
            if (!NS.empty())
              Signals.RelatedNamespaces[NS]++;
          }
        }
      },
      AST.getHeuristicResolver());
  return Signals;
}
} // namespace clangd
} // namespace clang
