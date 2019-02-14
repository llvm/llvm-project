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
#include "clang/AST/DeclBase.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/TypoCorrection.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <vector>

namespace clang {
namespace clangd {

namespace {

// Collects contexts visited during a Sema name lookup.
class VisitedContextCollector : public VisibleDeclConsumer {
public:
  void EnteredContext(DeclContext *Ctx) override { Visited.push_back(Ctx); }

  void FoundDecl(NamedDecl *ND, NamedDecl *Hiding, DeclContext *Ctx,
                 bool InBaseClass) override {}

  std::vector<DeclContext *> takeVisitedContexts() {
    return std::move(Visited);
  }

private:
  std::vector<DeclContext *> Visited;
};

} // namespace

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
    break;
  case diag::err_unknown_typename:
  case diag::err_unknown_typename_suggest:
  case diag::err_typename_nested_not_found:
  case diag::err_no_template:
  case diag::err_no_template_suggest:
  case diag::err_undeclared_use:
  case diag::err_undeclared_use_suggest:
  case diag::err_undeclared_var_use:
  case diag::err_undeclared_var_use_suggest:
  case diag::err_no_member: // Could be no member in namespace.
  case diag::err_no_member_suggest:
    if (LastUnresolvedName) {
      // Try to fix unresolved name caused by missing declaraion.
      // E.g.
      //   clang::SourceManager SM;
      //          ~~~~~~~~~~~~~
      //          UnresolvedName
      //   or
      //   namespace clang {  SourceManager SM; }
      //                      ~~~~~~~~~~~~~
      //                      UnresolvedName
      // We only attempt to recover a diagnostic if it has the same location as
      // the last seen unresolved name.
      if (DiagLevel >= DiagnosticsEngine::Error &&
          LastUnresolvedName->Loc == Info.getLocation())
        return fixUnresolvedName();
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
  return fixesForSymbols({*Matched});
}

std::vector<Fix>
IncludeFixer::fixesForSymbols(llvm::ArrayRef<Symbol> Syms) const {
  auto Inserted = [&](const Symbol &Sym, llvm::StringRef Header)
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
  // Deduplicate fixes by include headers. This doesn't distiguish symbols in
  // different scopes from the same header, but this case should be rare and is
  // thus ignored.
  llvm::StringSet<> InsertedHeaders;
  for (const auto &Sym : Syms) {
    for (const auto &Inc : getRankedIncludes(Sym)) {
      if (auto ToInclude = Inserted(Sym, Inc)) {
        if (ToInclude->second) {
          auto I = InsertedHeaders.try_emplace(ToInclude->first);
          if (!I.second)
            continue;
          if (auto Edit = Inserter->insert(ToInclude->first))
            Fixes.push_back(
                Fix{llvm::formatv("Add include {0} for symbol {1}{2}",
                                  ToInclude->first, Sym.Scope, Sym.Name),
                    {std::move(*Edit)}});
        }
      } else {
        vlog("Failed to calculate include insertion for {0} into {1}: {2}",
             File, Inc, ToInclude.takeError());
      }
    }
  }
  return Fixes;
}
class IncludeFixer::UnresolvedNameRecorder : public ExternalSemaSource {
public:
  UnresolvedNameRecorder(llvm::Optional<UnresolvedName> &LastUnresolvedName)
      : LastUnresolvedName(LastUnresolvedName) {}

  void InitializeSema(Sema &S) override { this->SemaPtr = &S; }

  // Captures the latest typo and treat it as an unresolved name that can
  // potentially be fixed by adding #includes.
  TypoCorrection CorrectTypo(const DeclarationNameInfo &Typo, int LookupKind,
                             Scope *S, CXXScopeSpec *SS,
                             CorrectionCandidateCallback &CCC,
                             DeclContext *MemberContext, bool EnteringContext,
                             const ObjCObjectPointerType *OPT) override {
    assert(SemaPtr && "Sema must have been set.");
    if (SemaPtr->isSFINAEContext())
      return TypoCorrection();
    if (!SemaPtr->SourceMgr.isWrittenInMainFile(Typo.getLoc()))
      return clang::TypoCorrection();

    // FIXME: support invalid scope before a type name. In the following
    // example, namespace "clang::tidy::" hasn't been declared/imported.
    //    namespace clang {
    //    void f() {
    //      tidy::Check c;
    //      ~~~~
    //      // or
    //      clang::tidy::Check c;
    //             ~~~~
    //    }
    //    }
    // For both cases, the typo and the diagnostic are both on "tidy", and no
    // diagnostic is generated for "Check". However, what we want to fix is
    // "clang::tidy::Check".

    // Extract the typed scope. This is not done lazily because `SS` can get
    // out of scope and it's relatively cheap.
    llvm::Optional<std::string> SpecifiedScope;
    if (SS && SS->isNotEmpty()) { // "::" or "ns::"
      if (auto *Nested = SS->getScopeRep()) {
        if (Nested->getKind() == NestedNameSpecifier::Global)
          SpecifiedScope = "";
        else if (const auto *NS = Nested->getAsNamespace())
          SpecifiedScope = printNamespaceScope(*NS);
        else
          // We don't fix symbols in scopes that are not top-level e.g. class
          // members, as we don't collect includes for them.
          return TypoCorrection();
      }
    }
    if (!SpecifiedScope && !S) // Give up if no scope available.
      return TypoCorrection();

    UnresolvedName Unresolved;
    Unresolved.Name = Typo.getAsString();
    Unresolved.Loc = Typo.getBeginLoc();

    auto *Sem = SemaPtr; // Avoid capturing `this`.
    Unresolved.GetScopes = [Sem, SpecifiedScope, S, LookupKind]() {
      std::vector<std::string> Scopes;
      if (SpecifiedScope) {
        Scopes.push_back(*SpecifiedScope);
      } else {
        assert(S);
        // No scope qualifier is specified. Collect all accessible scopes in the
        // context.
        VisitedContextCollector Collector;
        Sem->LookupVisibleDecls(
            S, static_cast<Sema::LookupNameKind>(LookupKind), Collector,
            /*IncludeGlobalScope=*/false,
            /*LoadExternal=*/false);

        Scopes.push_back("");
        for (const auto *Ctx : Collector.takeVisitedContexts())
          if (isa<NamespaceDecl>(Ctx))
            Scopes.push_back(printNamespaceScope(*Ctx));
      }
      return Scopes;
    };
    LastUnresolvedName = std::move(Unresolved);

    // Never return a valid correction to try to recover. Our suggested fixes
    // always require a rebuild.
    return TypoCorrection();
  }

private:
  Sema *SemaPtr = nullptr;

  llvm::Optional<UnresolvedName> &LastUnresolvedName;
};

llvm::IntrusiveRefCntPtr<ExternalSemaSource>
IncludeFixer::unresolvedNameRecorder() {
  return new UnresolvedNameRecorder(LastUnresolvedName);
}

std::vector<Fix> IncludeFixer::fixUnresolvedName() const {
  assert(LastUnresolvedName.hasValue());
  auto &Unresolved = *LastUnresolvedName;
  std::vector<std::string> Scopes = Unresolved.GetScopes();
  vlog("Trying to fix unresolved name \"{0}\" in scopes: [{1}]",
       Unresolved.Name, llvm::join(Scopes.begin(), Scopes.end(), ", "));

  FuzzyFindRequest Req;
  Req.AnyScope = false;
  Req.Query = Unresolved.Name;
  Req.Scopes = Scopes;
  Req.RestrictForCodeCompletion = true;
  Req.Limit = 100;

  SymbolSlab::Builder Matches;
  Index.fuzzyFind(Req, [&](const Symbol &Sym) {
    if (Sym.Name != Req.Query)
      return;
    if (!Sym.IncludeHeaders.empty())
      Matches.insert(Sym);
  });
  auto Syms = std::move(Matches).build();
  return fixesForSymbols(std::vector<Symbol>(Syms.begin(), Syms.end()));
}

} // namespace clangd
} // namespace clang
