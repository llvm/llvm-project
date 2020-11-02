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
#include "SourceCode.h"
#include "index/Index.h"
#include "index/Symbol.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/TypoCorrection.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
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
  switch (Info.getID()) {
  case diag::err_incomplete_nested_name_spec:
  case diag::err_incomplete_base_class:
  case diag::err_incomplete_member_access:
  case diag::err_incomplete_type:
  case diag::err_typecheck_decl_incomplete_type:
  case diag::err_typecheck_incomplete_tag:
  case diag::err_invalid_incomplete_type_use:
  case diag::err_sizeof_alignof_incomplete_or_sizeless_type:
  case diag::err_for_range_incomplete_type:
  case diag::err_func_def_incomplete_result:
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
      // Try to fix unresolved name caused by missing declaration.
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
  llvm::Optional<const SymbolSlab *> Symbols = lookupCached(ID);
  if (!Symbols)
    return {};
  const SymbolSlab &Syms = **Symbols;
  std::vector<Fix> Fixes;
  if (!Syms.empty()) {
    auto &Matched = *Syms.begin();
    if (!Matched.IncludeHeaders.empty() && Matched.Definition &&
        Matched.CanonicalDeclaration.FileURI == Matched.Definition.FileURI)
      Fixes = fixesForSymbols(Syms);
  }
  return Fixes;
}

std::vector<Fix> IncludeFixer::fixesForSymbols(const SymbolSlab &Syms) const {
  auto Inserted = [&](const Symbol &Sym, llvm::StringRef Header)
      -> llvm::Expected<std::pair<std::string, bool>> {
    auto ResolvedDeclaring =
        URI::resolve(Sym.CanonicalDeclaration.FileURI, File);
    if (!ResolvedDeclaring)
      return ResolvedDeclaring.takeError();
    auto ResolvedInserted = toHeaderFile(Header, File);
    if (!ResolvedInserted)
      return ResolvedInserted.takeError();
    auto Spelled = Inserter->calculateIncludePath(*ResolvedInserted, File);
    if (!Spelled)
      return error("Header not on include path");
    return std::make_pair(
        std::move(*Spelled),
        Inserter->shouldInsertInclude(*ResolvedDeclaring, *ResolvedInserted));
  };

  std::vector<Fix> Fixes;
  // Deduplicate fixes by include headers. This doesn't distinguish symbols in
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
            Fixes.push_back(Fix{std::string(llvm::formatv(
                                    "Add include {0} for symbol {1}{2}",
                                    ToInclude->first, Sym.Scope, Sym.Name)),
                                {std::move(*Edit)}});
        }
      } else {
        vlog("Failed to calculate include insertion for {0} into {1}: {2}", Inc,
             File, ToInclude.takeError());
      }
    }
  }
  return Fixes;
}

// Returns the identifiers qualified by an unresolved name. \p Loc is the
// start location of the unresolved name. For the example below, this returns
// "::X::Y" that is qualified by unresolved name "clangd":
//     clang::clangd::X::Y
//            ~
llvm::Optional<std::string> qualifiedByUnresolved(const SourceManager &SM,
                                                  SourceLocation Loc,
                                                  const LangOptions &LangOpts) {
  std::string Result;

  SourceLocation NextLoc = Loc;
  while (auto CCTok = Lexer::findNextToken(NextLoc, SM, LangOpts)) {
    if (!CCTok->is(tok::coloncolon))
      break;
    auto IDTok = Lexer::findNextToken(CCTok->getLocation(), SM, LangOpts);
    if (!IDTok || !IDTok->is(tok::raw_identifier))
      break;
    Result.append(("::" + IDTok->getRawIdentifier()).str());
    NextLoc = IDTok->getLocation();
  }
  if (Result.empty())
    return llvm::None;
  return Result;
}

// An unresolved name and its scope information that can be extracted cheaply.
struct CheapUnresolvedName {
  std::string Name;
  // This is the part of what was typed that was resolved, and it's in its
  // resolved form not its typed form (think `namespace clang { clangd::x }` -->
  // `clang::clangd::`).
  llvm::Optional<std::string> ResolvedScope;

  // Unresolved part of the scope. When the unresolved name is a specifier, we
  // use the name that comes after it as the alternative name to resolve and use
  // the specifier as the extra scope in the accessible scopes.
  llvm::Optional<std::string> UnresolvedScope;
};

// Extracts unresolved name and scope information around \p Unresolved.
// FIXME: try to merge this with the scope-wrangling code in CodeComplete.
llvm::Optional<CheapUnresolvedName> extractUnresolvedNameCheaply(
    const SourceManager &SM, const DeclarationNameInfo &Unresolved,
    CXXScopeSpec *SS, const LangOptions &LangOpts, bool UnresolvedIsSpecifier) {
  bool Invalid = false;
  llvm::StringRef Code = SM.getBufferData(
      SM.getDecomposedLoc(Unresolved.getBeginLoc()).first, &Invalid);
  if (Invalid)
    return llvm::None;
  CheapUnresolvedName Result;
  Result.Name = Unresolved.getAsString();
  if (SS && SS->isNotEmpty()) { // "::" or "ns::"
    if (auto *Nested = SS->getScopeRep()) {
      if (Nested->getKind() == NestedNameSpecifier::Global)
        Result.ResolvedScope = "";
      else if (const auto *NS = Nested->getAsNamespace()) {
        auto SpecifiedNS = printNamespaceScope(*NS);

        // Check the specifier spelled in the source.
        // If the resolved scope doesn't end with the spelled scope. The
        // resolved scope can come from a sema typo correction. For example,
        // sema assumes that "clangd::" is a typo of "clang::" and uses
        // "clang::" as the specified scope in:
        //     namespace clang { clangd::X; }
        // In this case, we use the "typo" specifier as extra scope instead
        // of using the scope assumed by sema.
        auto B = SM.getFileOffset(SS->getBeginLoc());
        auto E = SM.getFileOffset(SS->getEndLoc());
        std::string Spelling = (Code.substr(B, E - B) + "::").str();
        if (llvm::StringRef(SpecifiedNS).endswith(Spelling))
          Result.ResolvedScope = SpecifiedNS;
        else
          Result.UnresolvedScope = Spelling;
      } else if (const auto *ANS = Nested->getAsNamespaceAlias()) {
        Result.ResolvedScope = printNamespaceScope(*ANS->getNamespace());
      } else {
        // We don't fix symbols in scopes that are not top-level e.g. class
        // members, as we don't collect includes for them.
        return llvm::None;
      }
    }
  }

  if (UnresolvedIsSpecifier) {
    // If the unresolved name is a specifier e.g.
    //      clang::clangd::X
    //             ~~~~~~
    // We try to resolve clang::clangd::X instead of clang::clangd.
    // FIXME: We won't be able to fix include if the specifier is what we
    // should resolve (e.g. it's a class scope specifier). Collecting include
    // headers for nested types could make this work.

    // Not using the end location as it doesn't always point to the end of
    // identifier.
    if (auto QualifiedByUnresolved =
            qualifiedByUnresolved(SM, Unresolved.getBeginLoc(), LangOpts)) {
      auto Split = splitQualifiedName(*QualifiedByUnresolved);
      if (!Result.UnresolvedScope)
        Result.UnresolvedScope.emplace();
      // If UnresolvedSpecifiedScope is already set, we simply append the
      // extra scope. Suppose the unresolved name is "index" in the following
      // example:
      //   namespace clang {  clangd::index::X; }
      //                      ~~~~~~  ~~~~~
      // "clangd::" is assumed to be clang:: by Sema, and we would have used
      // it as extra scope. With "index" being a specifier, we append "index::"
      // to the extra scope.
      Result.UnresolvedScope->append((Result.Name + Split.first).str());
      Result.Name = std::string(Split.second);
    }
  }
  return Result;
}

/// Returns all namespace scopes that the unqualified lookup would visit.
std::vector<std::string>
collectAccessibleScopes(Sema &Sem, const DeclarationNameInfo &Typo, Scope *S,
                        Sema::LookupNameKind LookupKind) {
  std::vector<std::string> Scopes;
  VisitedContextCollector Collector;
  Sem.LookupVisibleDecls(S, LookupKind, Collector,
                         /*IncludeGlobalScope=*/false,
                         /*LoadExternal=*/false);

  Scopes.push_back("");
  for (const auto *Ctx : Collector.takeVisitedContexts()) {
    if (isa<NamespaceDecl>(Ctx))
      Scopes.push_back(printNamespaceScope(*Ctx));
  }
  return Scopes;
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
    if (!isInsideMainFile(Typo.getLoc(), SemaPtr->SourceMgr))
      return clang::TypoCorrection();

    auto Extracted = extractUnresolvedNameCheaply(
        SemaPtr->SourceMgr, Typo, SS, SemaPtr->LangOpts,
        static_cast<Sema::LookupNameKind>(LookupKind) ==
            Sema::LookupNameKind::LookupNestedNameSpecifierName);
    if (!Extracted)
      return TypoCorrection();

    UnresolvedName Unresolved;
    Unresolved.Name = Extracted->Name;
    Unresolved.Loc = Typo.getBeginLoc();
    if (!Extracted->ResolvedScope && !S) // Give up if no scope available.
      return TypoCorrection();

    if (Extracted->ResolvedScope)
      Unresolved.Scopes.push_back(*Extracted->ResolvedScope);
    else // no qualifier or qualifier is unresolved.
      Unresolved.Scopes = collectAccessibleScopes(
          *SemaPtr, Typo, S, static_cast<Sema::LookupNameKind>(LookupKind));

    if (Extracted->UnresolvedScope) {
      for (std::string &Scope : Unresolved.Scopes)
        Scope += *Extracted->UnresolvedScope;
    }

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
  vlog("Trying to fix unresolved name \"{0}\" in scopes: [{1}]",
       Unresolved.Name, llvm::join(Unresolved.Scopes, ", "));

  FuzzyFindRequest Req;
  Req.AnyScope = false;
  Req.Query = Unresolved.Name;
  Req.Scopes = Unresolved.Scopes;
  Req.RestrictForCodeCompletion = true;
  Req.Limit = 100;

  if (llvm::Optional<const SymbolSlab *> Syms = fuzzyFindCached(Req))
    return fixesForSymbols(**Syms);

  return {};
}

llvm::Optional<const SymbolSlab *>
IncludeFixer::fuzzyFindCached(const FuzzyFindRequest &Req) const {
  auto ReqStr = llvm::formatv("{0}", toJSON(Req)).str();
  auto I = FuzzyFindCache.find(ReqStr);
  if (I != FuzzyFindCache.end())
    return &I->second;

  if (IndexRequestCount >= IndexRequestLimit)
    return llvm::None;
  IndexRequestCount++;

  SymbolSlab::Builder Matches;
  Index.fuzzyFind(Req, [&](const Symbol &Sym) {
    if (Sym.Name != Req.Query)
      return;
    if (!Sym.IncludeHeaders.empty())
      Matches.insert(Sym);
  });
  auto Syms = std::move(Matches).build();
  auto E = FuzzyFindCache.try_emplace(ReqStr, std::move(Syms));
  return &E.first->second;
}

llvm::Optional<const SymbolSlab *>
IncludeFixer::lookupCached(const SymbolID &ID) const {
  LookupRequest Req;
  Req.IDs.insert(ID);

  auto I = LookupCache.find(ID);
  if (I != LookupCache.end())
    return &I->second;

  if (IndexRequestCount >= IndexRequestLimit)
    return llvm::None;
  IndexRequestCount++;

  // FIXME: consider batching the requests for all diagnostics.
  SymbolSlab::Builder Matches;
  Index.lookup(Req, [&](const Symbol &Sym) { Matches.insert(Sym); });
  auto Syms = std::move(Matches).build();

  std::vector<Fix> Fixes;
  if (!Syms.empty()) {
    auto &Matched = *Syms.begin();
    if (!Matched.IncludeHeaders.empty() && Matched.Definition &&
        Matched.CanonicalDeclaration.FileURI == Matched.Definition.FileURI)
      Fixes = fixesForSymbols(Syms);
  }
  auto E = LookupCache.try_emplace(ID, std::move(Syms));
  return &E.first->second;
}

} // namespace clangd
} // namespace clang
