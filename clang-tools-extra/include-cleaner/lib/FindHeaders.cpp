//===--- FindHeaders.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "clang-include-cleaner/Record.h"
#include "clang/Basic/SourceManager.h"

namespace clang::include_cleaner {

llvm::SmallVector<Header> findHeaders(const SymbolLocation &Loc,
                                      const SourceManager &SM,
                                      const PragmaIncludes *PI) {
  llvm::SmallVector<Header> Results;
  switch (Loc.kind()) {
  case SymbolLocation::Physical: {
    FileID FID = SM.getFileID(SM.getExpansionLoc(Loc.physical()));
    const FileEntry *FE = SM.getFileEntryForID(FID);
    if (!PI) {
      return FE ? llvm::SmallVector<Header>{Header(FE)}
                : llvm::SmallVector<Header>();
    }
    while (FE) {
      Results.push_back(Header(FE));
      // FIXME: compute transitive exporter headers.
      for (const auto *Export : PI->getExporters(FE, SM.getFileManager()))
        Results.push_back(Header(Export));

      llvm::StringRef VerbatimSpelling = PI->getPublic(FE);
      if (!VerbatimSpelling.empty()) {
        Results.push_back(VerbatimSpelling);
        break;
      }
      if (PI->isSelfContained(FE) || FID == SM.getMainFileID())
        break;

      // Walkup the include stack for non self-contained headers.
      FID = SM.getDecomposedIncludedLoc(FID).first;
      FE = SM.getFileEntryForID(FID);
    }
    return Results;
  }
  case SymbolLocation::Standard: {
    for (const auto &H : Loc.standard().headers()) {
      Results.push_back(H);
      for (const auto *Export : PI->getExporters(H, SM.getFileManager()))
        Results.push_back(Header(Export));
    }
    return Results;
  }
  }
  llvm_unreachable("unhandled SymbolLocation kind!");
}

} // namespace clang::include_cleaner
