//===--- FindHeaders.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "TypesInternal.h"
#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/FileEntry.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>
#include <utility>

namespace clang::include_cleaner {
namespace {
llvm::SmallVector<Hinted<Header>>
applyHints(llvm::SmallVector<Hinted<Header>> Headers, Hints H) {
  for (auto &Header : Headers)
    Header.Hint |= H;
  return Headers;
}

llvm::SmallVector<Header> ranked(llvm::SmallVector<Hinted<Header>> Headers) {
  llvm::stable_sort(llvm::reverse(Headers),
                    [](const Hinted<Header> &LHS, const Hinted<Header> &RHS) {
                      return LHS < RHS;
                    });
  return llvm::SmallVector<Header>(Headers.begin(), Headers.end());
}

// Return the basename from a verbatim header spelling, leaves only the file
// name.
llvm::StringRef basename(llvm::StringRef Header) {
  Header = Header.trim("<>\"");
  if (auto LastSlash = Header.rfind('/'); LastSlash != Header.npos)
    Header = Header.drop_front(LastSlash + 1);
  // Drop everything after first `.` (dot).
  // foo.h -> foo
  // foo.cu.h -> foo
  Header = Header.substr(0, Header.find('.'));
  return Header;
}

// Check if spelling of \p H matches \p DeclName.
bool nameMatch(llvm::StringRef DeclName, Header H) {
  switch (H.kind()) {
  case Header::Physical:
    return basename(H.physical()->getName()).equals_insensitive(DeclName);
  case Header::Standard:
    return basename(H.standard().name()).equals_insensitive(DeclName);
  case Header::Verbatim:
    return basename(H.verbatim()).equals_insensitive(DeclName);
  }
  llvm_unreachable("unhandled Header kind!");
}

llvm::StringRef symbolName(const Symbol &S) {
  switch (S.kind()) {
  case Symbol::Declaration:
    // Unnamed decls like operators and anonymous structs won't get any name
    // match.
    if (const auto *ND = llvm::dyn_cast<NamedDecl>(&S.declaration()))
      if (auto *II = ND->getIdentifier())
        return II->getName();
    return "";
  case Symbol::Macro:
    return S.macro().Name->getName();
  }
  llvm_unreachable("unhandled Symbol kind!");
}

} // namespace

llvm::SmallVector<Hinted<Header>> findHeaders(const SymbolLocation &Loc,
                                              const SourceManager &SM,
                                              const PragmaIncludes *PI) {
  auto IsPublicHeader = [&PI](const FileEntry *FE) {
    return (PI->isPrivate(FE) || !PI->isSelfContained(FE))
               ? Hints::None
               : Hints::PublicHeader;
  };
  llvm::SmallVector<Hinted<Header>> Results;
  switch (Loc.kind()) {
  case SymbolLocation::Physical: {
    FileID FID = SM.getFileID(SM.getExpansionLoc(Loc.physical()));
    const FileEntry *FE = SM.getFileEntryForID(FID);
    if (!FE)
      return {};
    if (!PI)
      return {{FE, Hints::PublicHeader}};
    while (FE) {
      Hints CurrentHints = IsPublicHeader(FE);
      Results.emplace_back(FE, CurrentHints);
      // FIXME: compute transitive exporter headers.
      for (const auto *Export : PI->getExporters(FE, SM.getFileManager()))
        Results.emplace_back(Export, IsPublicHeader(Export));

      if (auto Verbatim = PI->getPublic(FE); !Verbatim.empty()) {
        Results.emplace_back(Verbatim,
                             Hints::PublicHeader | Hints::PreferredHeader);
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
      Results.emplace_back(H, Hints::PublicHeader);
      for (const auto *Export : PI->getExporters(H, SM.getFileManager()))
        Results.emplace_back(Header(Export), IsPublicHeader(Export));
    }
    // StandardLibrary returns headers in preference order, so only mark the
    // first.
    if (!Results.empty())
      Results.front().Hint |= Hints::PreferredHeader;
    return Results;
  }
  }
  llvm_unreachable("unhandled SymbolLocation kind!");
}

llvm::SmallVector<Header> headersForSymbol(const Symbol &S,
                                           const SourceManager &SM,
                                           const PragmaIncludes *PI) {
  // Get headers for all the locations providing Symbol. Same header can be
  // reached through different traversals, deduplicate those into a single
  // Header by merging their hints.
  llvm::SmallVector<Hinted<Header>> Headers;
  for (auto &Loc : locateSymbol(S))
    Headers.append(applyHints(findHeaders(Loc, SM, PI), Loc.Hint));
  // If two Headers probably refer to the same file (e.g. Verbatim(foo.h) and
  // Physical(/path/to/foo.h), we won't deduplicate them or merge their hints
  llvm::stable_sort(
      Headers, [](const Hinted<Header> &LHS, const Hinted<Header> &RHS) {
        return static_cast<Header>(LHS) < static_cast<Header>(RHS);
      });
  auto *Write = Headers.begin();
  for (auto *Read = Headers.begin(); Read != Headers.end(); ++Write) {
    *Write = *Read++;
    while (Read != Headers.end() &&
           static_cast<Header>(*Write) == static_cast<Header>(*Read)) {
      Write->Hint |= Read->Hint;
      ++Read;
    }
  }
  Headers.erase(Write, Headers.end());

  // Add name match hints to deduplicated providers.
  llvm::StringRef SymbolName = symbolName(S);
  for (auto &H : Headers) {
    if (nameMatch(SymbolName, H))
      H.Hint |= Hints::PreferredHeader;
  }

  // FIXME: Introduce a MainFile header kind or signal and boost it.
  return ranked(std::move(Headers));
}
} // namespace clang::include_cleaner
