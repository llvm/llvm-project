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
                                      const PragmaIncludes &PI) {
  llvm::SmallVector<Header> Results;
  switch (Loc.kind()) {
  case SymbolLocation::Physical: {
    // FIXME: Handle non self-contained files.
    FileID FID = SM.getFileID(Loc.physical());
    const auto *FE = SM.getFileEntryForID(FID);
    if (!FE)
      return {};

    // We treat the spelling header in the IWYU pragma as the final public
    // header.
    // FIXME: look for exporters if the public header is exported by another
    // header.
    llvm::StringRef VerbatimSpelling = PI.getPublic(FE);
    if (!VerbatimSpelling.empty())
      return {Header(VerbatimSpelling)};

    Results = {Header(FE)};
    // FIXME: compute transitive exporter headers.
    for (const auto *Export : PI.getExporters(FE, SM.getFileManager()))
      Results.push_back(Export);
    return Results;
  }
  case SymbolLocation::Standard: {
    for (const auto &H : Loc.standard().headers())
      Results.push_back(H);
    return Results;
  }
  }
  llvm_unreachable("unhandled SymbolLocation kind!");
}

} // namespace clang::include_cleaner