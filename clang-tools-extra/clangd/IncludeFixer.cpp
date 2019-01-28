//===--- IncludeFixer.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncludeFixer.h"
#include "AST.h"
#include "Diagnostics.h"
#include "Logger.h"
#include "SourceCode.h"
#include "Trace.h"
#include "index/Index.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "llvm/ADT/None.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

namespace clang {
namespace clangd {

std::vector<Fix> IncludeFixer::fix(DiagnosticsEngine::Level DiagLevel,
                                   const clang::Diagnostic &Info) const {
  if (IndexRequestCount >= IndexRequestLimit)
    return {}; // Avoid querying index too many times in a single parse.
  switch (Info.getID()) {
  case diag::err_incomplete_type:
  case diag::err_incomplete_member_access:
  case diag::err_incomplete_base_class:
    // Incomplete type diagnostics should have a QualType argument for the
    // incomplete type.
    for (unsigned Idx = 0; Idx < Info.getNumArgs(); ++Idx) {
      if (Info.getArgKind(Idx) == DiagnosticsEngine::ak_qualtype) {
        auto QT = QualType::getFromOpaquePtr((void *)Info.getRawArg(Idx));
        if (const Type *T = QT.getTypePtrOrNull())
          if (T->isIncompleteType())
            return fixIncompleteType(*T);
      }
    }
  }
  return {};
}

std::vector<Fix> IncludeFixer::fixIncompleteType(const Type &T) const {
  // Only handle incomplete TagDecl type.
  const TagDecl *TD = T.getAsTagDecl();
  if (!TD)
    return {};
  std::string TypeName = printQualifiedName(*TD);
  trace::Span Tracer("Fix include for incomplete type");
  SPAN_ATTACH(Tracer, "type", TypeName);
  vlog("Trying to fix include for incomplete type {0}", TypeName);

  auto ID = getSymbolID(TD);
  if (!ID)
    return {};
  ++IndexRequestCount;
  // FIXME: consider batching the requests for all diagnostics.
  // FIXME: consider caching the lookup results.
  LookupRequest Req;
  Req.IDs.insert(*ID);
  llvm::Optional<Symbol> Matched;
  Index.lookup(Req, [&](const Symbol &Sym) {
    if (Matched)
      return;
    Matched = Sym;
  });

  if (!Matched || Matched->IncludeHeaders.empty() || !Matched->Definition ||
      Matched->CanonicalDeclaration.FileURI != Matched->Definition.FileURI)
    return {};
  return fixesForSymbol(*Matched);
}

std::vector<Fix> IncludeFixer::fixesForSymbol(const Symbol &Sym) const {
  auto Inserted = [&](llvm::StringRef Header)
      -> llvm::Expected<std::pair<std::string, bool>> {
    auto ResolvedDeclaring =
        toHeaderFile(Sym.CanonicalDeclaration.FileURI, File);
    if (!ResolvedDeclaring)
      return ResolvedDeclaring.takeError();
    auto ResolvedInserted = toHeaderFile(Header, File);
    if (!ResolvedInserted)
      return ResolvedInserted.takeError();
    return std::make_pair(
        Inserter->calculateIncludePath(*ResolvedDeclaring, *ResolvedInserted),
        Inserter->shouldInsertInclude(*ResolvedDeclaring, *ResolvedInserted));
  };

  std::vector<Fix> Fixes;
  for (const auto &Inc : getRankedIncludes(Sym)) {
    if (auto ToInclude = Inserted(Inc)) {
      if (ToInclude->second)
        if (auto Edit = Inserter->insert(ToInclude->first))
          Fixes.push_back(
              Fix{llvm::formatv("Add include {0} for symbol {1}{2}",
                                ToInclude->first, Sym.Scope, Sym.Name),
                  {std::move(*Edit)}});
    } else {
      vlog("Failed to calculate include insertion for {0} into {1}: {2}", File,
           Inc, ToInclude.takeError());
    }
  }
  return Fixes;
}

} // namespace clangd
} // namespace clang
