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
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/FileEntry.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>
#include <queue>
#include <set>
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
  Header = llvm::sys::path::filename(Header);
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
    return basename(H.physical().getName()).equals_insensitive(DeclName);
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

Hints isPublicHeader(const FileEntry *FE, const PragmaIncludes &PI) {
  if (PI.isPrivate(FE) || !PI.isSelfContained(FE))
    return Hints::None;
  return Hints::PublicHeader;
}

llvm::SmallVector<Hinted<Header>>
hintedHeadersForStdHeaders(llvm::ArrayRef<tooling::stdlib::Header> Headers,
                           const SourceManager &SM, const PragmaIncludes *PI) {
  llvm::SmallVector<Hinted<Header>> Results;
  for (const auto &H : Headers) {
    Results.emplace_back(H, Hints::PublicHeader | Hints::OriginHeader);
    if (!PI)
      continue;
    for (FileEntryRef Export : PI->getExporters(H, SM.getFileManager()))
      Results.emplace_back(Header(Export), isPublicHeader(Export, *PI));
  }
  // StandardLibrary returns headers in preference order, so only mark the
  // first.
  if (!Results.empty())
    Results.front().Hint |= Hints::PreferredHeader;
  return Results;
}

// Symbol to header mapping for std::move and std::remove, based on number of
// parameters.
std::optional<tooling::stdlib::Header>
headerForAmbiguousStdSymbol(const NamedDecl *ND) {
  if (!ND->isInStdNamespace())
    return {};
  if (auto* USD = llvm::dyn_cast<UsingShadowDecl>(ND))
    ND = USD->getTargetDecl();
  const auto *FD = ND->getAsFunction();
  if (!FD)
    return std::nullopt;
  llvm::StringRef FName = symbolName(*ND);
  if (FName == "move") {
    if (FD->getNumParams() == 1)
      // move(T&& t)
      return tooling::stdlib::Header::named("<utility>");
    if (FD->getNumParams() == 3 || FD->getNumParams() == 4)
      // move(InputIt first, InputIt last, OutputIt dest);
      // move(ExecutionPolicy&& policy, ForwardIt1 first,
      // ForwardIt1 last, ForwardIt2 d_first);
      return tooling::stdlib::Header::named("<algorithm>");
  } else if (FName == "remove") {
    if (FD->getNumParams() == 1)
      // remove(const char*);
      return tooling::stdlib::Header::named("<cstdio>");
    if (FD->getNumParams() == 3)
      // remove(ForwardIt first, ForwardIt last, const T& value);
      return tooling::stdlib::Header::named("<algorithm>");
  }
  return std::nullopt;
}

// Special-case symbols without proper locations, like the ambiguous standard
// library symbols (e.g. std::move) or builtin declarations.
std::optional<llvm::SmallVector<Hinted<Header>>>
headersForSpecialSymbol(const Symbol &S, const SourceManager &SM,
                        const PragmaIncludes *PI) {
  // Our special casing logic only deals with decls, so bail out early for
  // macros.
  if (S.kind() != Symbol::Declaration)
    return std::nullopt;
  const auto *ND = llvm::cast<NamedDecl>(&S.declaration());
  // We map based on names, so again bail out early if there are no names.
  if (!ND)
    return std::nullopt;
  auto *II = ND->getIdentifier();
  if (!II)
    return std::nullopt;

  // Check first for symbols that are part of our stdlib mapping. As we have
  // header names for those.
  if (auto Header = headerForAmbiguousStdSymbol(ND)) {
    return applyHints(hintedHeadersForStdHeaders({*Header}, SM, PI),
                      Hints::CompleteSymbol);
  }

  // Now check for builtin symbols, we shouldn't suggest any headers for ones
  // without any headers.
  if (auto ID = II->getBuiltinID()) {
    const char *BuiltinHeader =
        ND->getASTContext().BuiltinInfo.getHeaderName(ID);
    if (!BuiltinHeader)
      return llvm::SmallVector<Hinted<Header>>{};
    // FIXME: Use the header mapping for builtins with a known header.
  }
  return std::nullopt;
}

} // namespace

llvm::SmallVector<Hinted<Header>> findHeaders(const SymbolLocation &Loc,
                                              const SourceManager &SM,
                                              const PragmaIncludes *PI) {
  llvm::SmallVector<Hinted<Header>> Results;
  switch (Loc.kind()) {
  case SymbolLocation::Physical: {
    FileID FID = SM.getFileID(SM.getExpansionLoc(Loc.physical()));
    OptionalFileEntryRef FE = SM.getFileEntryRefForID(FID);
    if (!FE)
      return {};
    if (!PI)
      return {{*FE, Hints::PublicHeader | Hints::OriginHeader}};
    bool IsOrigin = true;
    std::queue<FileEntryRef> Exporters;
    while (FE) {
      Results.emplace_back(*FE,
                           isPublicHeader(*FE, *PI) |
                               (IsOrigin ? Hints::OriginHeader : Hints::None));
      for (FileEntryRef Export : PI->getExporters(*FE, SM.getFileManager()))
        Exporters.push(Export);

      if (auto Verbatim = PI->getPublic(*FE); !Verbatim.empty()) {
        Results.emplace_back(Verbatim,
                             Hints::PublicHeader | Hints::PreferredHeader);
        break;
      }
      if (PI->isSelfContained(*FE) || FID == SM.getMainFileID())
        break;

      // Walkup the include stack for non self-contained headers.
      FID = SM.getDecomposedIncludedLoc(FID).first;
      FE = SM.getFileEntryRefForID(FID);
      IsOrigin = false;
    }
    // Now traverse provider trees rooted at exporters.
    // Note that we only traverse export edges, and ignore private -> public
    // mappings, as those pragmas apply to exporter, and not the main provider
    // being exported in this header.
    std::set<const FileEntry *> SeenExports;
    while (!Exporters.empty()) {
      FileEntryRef Export = Exporters.front();
      Exporters.pop();
      if (!SeenExports.insert(Export).second) // In case of cyclic exports
        continue;
      Results.emplace_back(Export, isPublicHeader(Export, *PI));
      for (FileEntryRef Export : PI->getExporters(Export, SM.getFileManager()))
        Exporters.push(Export);
    }
    return Results;
  }
  case SymbolLocation::Standard: {
    return hintedHeadersForStdHeaders(Loc.standard().headers(), SM, PI);
  }
  }
  llvm_unreachable("unhandled SymbolLocation kind!");
}

llvm::SmallVector<Header> headersForSymbol(const Symbol &S,
                                           const Preprocessor &PP,
                                           const PragmaIncludes *PI) {
  const auto &SM = PP.getSourceManager();
  // Get headers for all the locations providing Symbol. Same header can be
  // reached through different traversals, deduplicate those into a single
  // Header by merging their hints.
  llvm::SmallVector<Hinted<Header>> Headers;
  if (auto SpecialHeaders = headersForSpecialSymbol(S, SM, PI)) {
    Headers = std::move(*SpecialHeaders);
  } else {
    for (auto &Loc : locateSymbol(S, PP.getLangOpts()))
      Headers.append(applyHints(findHeaders(Loc, SM, PI), Loc.Hint));
  }
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
    // Don't apply name match hints to standard headers as the standard headers
    // are already ranked in the stdlib mapping.
    if (H.kind() == Header::Standard)
      continue;
    // Don't apply name match hints to exporting headers. As they usually have
    // names similar to the original header, e.g. foo_wrapper/foo.h vs
    // foo/foo.h, but shouldn't be preferred (unless marked as the public
    // interface).
    if ((H.Hint & Hints::OriginHeader) == Hints::None)
      continue;
    if (nameMatch(SymbolName, H))
      H.Hint |= Hints::PreferredHeader;
  }

  // FIXME: Introduce a MainFile header kind or signal and boost it.
  return ranked(std::move(Headers));
}
} // namespace clang::include_cleaner
