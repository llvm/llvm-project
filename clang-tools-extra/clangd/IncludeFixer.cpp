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
#include "clang/Basic/DiagnosticParse.h"
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace {

llvm::Optional<llvm::StringRef> getArgStr(const clang::Diagnostic &Info,
                                          unsigned Index) {
  switch (Info.getArgKind(Index)) {
  case DiagnosticsEngine::ak_c_string:
    return llvm::StringRef(Info.getArgCStr(Index));
  case DiagnosticsEngine::ak_std_string:
    return llvm::StringRef(Info.getArgStdStr(Index));
  default:
    return llvm::None;
  }
}

std::vector<Fix> only(llvm::Optional<Fix> F) {
  if (F)
    return {std::move(*F)};
  return {};
}

} // namespace

std::vector<Fix> IncludeFixer::fix(DiagnosticsEngine::Level DiagLevel,
                                   const clang::Diagnostic &Info) const {
  switch (Info.getID()) {
  /*
   There are many "incomplete type" diagnostics!
   They are almost all Sema diagnostics with "incomplete" in the name.

   sed -n '/CLASS_NOTE/! s/DIAG(\\([^,]*\\).*)/  case diag::\\1:/p' \
     tools/clang/include/clang/Basic/DiagnosticSemaKinds.inc | grep incomplete
  */
  // clang-format off
  //case diag::err_alignof_member_of_incomplete_type:
  case diag::err_array_incomplete_or_sizeless_type:
  case diag::err_array_size_incomplete_type:
  case diag::err_asm_incomplete_type:
  case diag::err_assoc_type_incomplete:
  case diag::err_bad_cast_incomplete:
  case diag::err_call_function_incomplete_return:
  case diag::err_call_incomplete_argument:
  case diag::err_call_incomplete_return:
  case diag::err_capture_of_incomplete_or_sizeless_type:
  case diag::err_catch_incomplete:
  case diag::err_catch_incomplete_ptr:
  case diag::err_catch_incomplete_ref:
  case diag::err_cconv_incomplete_param_type:
  case diag::err_coroutine_promise_type_incomplete:
  case diag::err_covariant_return_incomplete:
  //case diag::err_deduced_class_template_incomplete:
  case diag::err_delete_incomplete_class_type:
  case diag::err_dereference_incomplete_type:
  case diag::err_exception_spec_incomplete_type:
  case diag::err_field_incomplete_or_sizeless:
  case diag::err_for_range_incomplete_type:
  case diag::err_func_def_incomplete_result:
  case diag::err_ice_incomplete_type:
  case diag::err_illegal_message_expr_incomplete_type:
  case diag::err_incomplete_base_class:
  case diag::err_incomplete_enum:
  case diag::err_incomplete_in_exception_spec:
  case diag::err_incomplete_member_access:
  case diag::err_incomplete_nested_name_spec:
  case diag::err_incomplete_object_call:
  case diag::err_incomplete_receiver_type:
  case diag::err_incomplete_synthesized_property:
  case diag::err_incomplete_type:
  case diag::err_incomplete_type_objc_at_encode:
  case diag::err_incomplete_type_used_in_type_trait_expr:
  case diag::err_incomplete_typeid:
  case diag::err_init_incomplete_type:
  case diag::err_invalid_incomplete_type_use:
  case diag::err_lambda_incomplete_result:
  //case diag::err_matrix_incomplete_index:
  //case diag::err_matrix_separate_incomplete_index:
  case diag::err_memptr_incomplete:
  case diag::err_new_incomplete_or_sizeless_type:
  case diag::err_objc_incomplete_boxed_expression_type:
  case diag::err_objc_index_incomplete_class_type:
  case diag::err_offsetof_incomplete_type:
  case diag::err_omp_firstprivate_incomplete_type:
  case diag::err_omp_incomplete_type:
  case diag::err_omp_lastprivate_incomplete_type:
  case diag::err_omp_linear_incomplete_type:
  case diag::err_omp_private_incomplete_type:
  case diag::err_omp_reduction_incomplete_type:
  case diag::err_omp_section_incomplete_type:
  case diag::err_omp_threadprivate_incomplete_type:
  case diag::err_second_parameter_to_va_arg_incomplete:
  case diag::err_sizeof_alignof_incomplete_or_sizeless_type:
  case diag::err_subscript_incomplete_or_sizeless_type:
  case diag::err_switch_incomplete_class_type:
  case diag::err_temp_copy_incomplete:
  //case diag::err_template_arg_deduced_incomplete_pack:
  case diag::err_template_nontype_parm_incomplete:
  //case diag::err_tentative_def_incomplete_type:
  case diag::err_throw_incomplete:
  case diag::err_throw_incomplete_ptr:
  case diag::err_typecheck_arithmetic_incomplete_or_sizeless_type:
  case diag::err_typecheck_cast_to_incomplete:
  case diag::err_typecheck_decl_incomplete_type:
  //case diag::err_typecheck_incomplete_array_needs_initializer:
  case diag::err_typecheck_incomplete_tag:
  case diag::err_typecheck_incomplete_type_not_modifiable_lvalue:
  case diag::err_typecheck_nonviable_condition_incomplete:
  case diag::err_underlying_type_of_incomplete_enum:
  case diag::ext_incomplete_in_exception_spec:
  //case diag::ext_typecheck_compare_complete_incomplete_pointers:
  case diag::ext_typecheck_decl_incomplete_type:
  case diag::warn_delete_incomplete:
  case diag::warn_incomplete_encoded_type:
  //case diag::warn_printf_incomplete_specifier:
  case diag::warn_return_value_udt_incomplete:
  //case diag::warn_scanf_scanlist_incomplete:
  //case diag::warn_tentative_incomplete_array:
    //  clang-format on
    // Incomplete type diagnostics should have a QualType argument for the
    // incomplete type.
    for (unsigned Idx = 0; Idx < Info.getNumArgs(); ++Idx) {
      if (Info.getArgKind(Idx) == DiagnosticsEngine::ak_qualtype) {
        auto QT = QualType::getFromOpaquePtr((void *)Info.getRawArg(Idx));
        if (const Type *T = QT.getTypePtrOrNull()) {
          if (T->isIncompleteType())
            return fixIncompleteType(*T);
          // `enum x : int;' is not formally an incomplete type.
          // We may need a full definition anyway.
          if (auto * ET = llvm::dyn_cast<EnumType>(T))
            if (!ET->getDecl()->getDefinition())
              return fixIncompleteType(*T);
        }
      }
    }
    break;

  case diag::err_unknown_typename:
  case diag::err_unknown_typename_suggest:
  case diag::err_unknown_type_or_class_name_suggest:
  case diag::err_expected_class_name:
  case diag::err_typename_nested_not_found:
  case diag::err_no_template:
  case diag::err_no_template_suggest:
  case diag::err_undeclared_use:
  case diag::err_undeclared_use_suggest:
  case diag::err_undeclared_var_use:
  case diag::err_undeclared_var_use_suggest:
  case diag::err_no_member: // Could be no member in namespace.
  case diag::err_no_member_suggest:
  case diag::err_no_member_template:
  case diag::err_no_member_template_suggest:
  case diag::warn_implicit_function_decl:
  case diag::ext_implicit_function_decl_c99:
    dlog("Unresolved name at {0}, last typo was {1}",
         Info.getLocation().printToString(Info.getSourceManager()),
         LastUnresolvedName
             ? LastUnresolvedName->Loc.printToString(Info.getSourceManager())
             : "none");
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
      if (LastUnresolvedName->Loc == Info.getLocation())
        return fixUnresolvedName();
    }
    break;

  // Cases where clang explicitly knows which header to include.
  // (There's no fix provided for boring formatting reasons).
  case diag::err_implied_std_initializer_list_not_found:
    return only(insertHeader("<initializer_list>"));
  case diag::err_need_header_before_typeid:
    return only(insertHeader("<typeid>"));
  case diag::err_need_header_before_placement_new:
  case diag::err_implicit_coroutine_std_nothrow_type_not_found:
    return only(insertHeader("<new>"));
  case diag::err_omp_implied_type_not_found:
  case diag::err_omp_interop_type_not_found:
    return only(insertHeader("<omp.h>"));
  case diag::err_implied_coroutine_type_not_found:
    return only(insertHeader("<coroutine>"));
  case diag::err_implied_comparison_category_type_not_found:
    return only(insertHeader("<compare>"));
  case diag::note_include_header_or_declare:
    if (Info.getNumArgs() > 0)
      if (auto Header = getArgStr(Info, 0))
        return only(insertHeader(("<" + *Header + ">").str(),
                                 getArgStr(Info, 1).value_or("")));
    break;
  }

  return {};
}

llvm::Optional<Fix> IncludeFixer::insertHeader(llvm::StringRef Spelled,
                                               llvm::StringRef Symbol) const {
  Fix F;

  if (auto Edit = Inserter->insert(Spelled))
    F.Edits.push_back(std::move(*Edit));
  else
    return llvm::None;

  if (Symbol.empty())
    F.Message = llvm::formatv("Include {0}", Spelled);
  else
    F.Message = llvm::formatv("Include {0} for symbol {1}", Spelled, Symbol);

  return F;
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
          if (!InsertedHeaders.try_emplace(ToInclude->first).second)
            continue;
          if (auto Fix =
                  insertHeader(ToInclude->first, (Sym.Scope + Sym.Name).str()))
            Fixes.push_back(std::move(*Fix));
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
  // Accept qualifier written within macro arguments, but not macro bodies.
  SourceLocation NextLoc = SM.getTopMacroCallerLoc(Loc);
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

llvm::Optional<std::string> getSpelledSpecifier(const CXXScopeSpec &SS,
    const SourceManager &SM) {
  // Support specifiers written within a single macro argument.
  if (!SM.isWrittenInSameFile(SS.getBeginLoc(), SS.getEndLoc()))
    return llvm::None;
  SourceRange Range(SM.getTopMacroCallerLoc(SS.getBeginLoc()), SM.getTopMacroCallerLoc(SS.getEndLoc()));
  if (Range.getBegin().isMacroID() || Range.getEnd().isMacroID())
    return llvm::None;

  return (toSourceCode(SM, Range) + "::").str();
}

// Extracts unresolved name and scope information around \p Unresolved.
// FIXME: try to merge this with the scope-wrangling code in CodeComplete.
llvm::Optional<CheapUnresolvedName> extractUnresolvedNameCheaply(
    const SourceManager &SM, const DeclarationNameInfo &Unresolved,
    CXXScopeSpec *SS, const LangOptions &LangOpts, bool UnresolvedIsSpecifier) {
  CheapUnresolvedName Result;
  Result.Name = Unresolved.getAsString();
  if (SS && SS->isNotEmpty()) { // "::" or "ns::"
    if (auto *Nested = SS->getScopeRep()) {
      if (Nested->getKind() == NestedNameSpecifier::Global) {
        Result.ResolvedScope = "";
      } else if (const auto *NS = Nested->getAsNamespace()) {
        std::string SpecifiedNS = printNamespaceScope(*NS);
        llvm::Optional<std::string> Spelling = getSpelledSpecifier(*SS, SM);

        // Check the specifier spelled in the source.
        // If the resolved scope doesn't end with the spelled scope, the
        // resolved scope may come from a sema typo correction. For example,
        // sema assumes that "clangd::" is a typo of "clang::" and uses
        // "clang::" as the specified scope in:
        //     namespace clang { clangd::X; }
        // In this case, we use the "typo" specifier as extra scope instead
        // of using the scope assumed by sema.
        if (!Spelling || llvm::StringRef(SpecifiedNS).endswith(*Spelling)) {
          Result.ResolvedScope = std::move(SpecifiedNS);
        } else {
          Result.UnresolvedScope = std::move(*Spelling);
        }
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
  // Collects contexts visited during a Sema name lookup.
  struct VisitedContextCollector : public VisibleDeclConsumer {
    VisitedContextCollector(std::vector<std::string> &Out) : Out(Out) {}
    void EnteredContext(DeclContext *Ctx) override {
      if (llvm::isa<NamespaceDecl>(Ctx))
        Out.push_back(printNamespaceScope(*Ctx));
    }
    void FoundDecl(NamedDecl *ND, NamedDecl *Hiding, DeclContext *Ctx,
                   bool InBaseClass) override {}
    std::vector<std::string> &Out;
  };

  std::vector<std::string> Scopes;
  Scopes.push_back("");
  VisitedContextCollector Collector(Scopes);
  Sem.LookupVisibleDecls(S, LookupKind, Collector,
                         /*IncludeGlobalScope=*/false,
                         /*LoadExternal=*/false);
  llvm::sort(Scopes);
  Scopes.erase(std::unique(Scopes.begin(), Scopes.end()), Scopes.end());
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
    dlog("CorrectTypo: {0}", Typo.getAsString());
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
  assert(LastUnresolvedName);
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
