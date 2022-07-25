//===--- CodeComplete.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code completion has several moving parts:
//  - AST-based completions are provided using the completion hooks in Sema.
//  - external completions are retrieved from the index (using hints from Sema)
//  - the two sources overlap, and must be merged and overloads bundled
//  - results must be scored and ranked (see Quality.h) before rendering
//
// Signature help works in a similar way as code completion, but it is simpler:
// it's purely AST-based, and there are few candidates.
//
//===----------------------------------------------------------------------===//

#include "CodeComplete.h"
#include "AST.h"
#include "CodeCompletionStrings.h"
#include "Compiler.h"
#include "ExpectedTypes.h"
#include "FileDistance.h"
#include "FuzzyMatch.h"
#include "Headers.h"
#include "Hover.h"
#include "Preamble.h"
#include "Protocol.h"
#include "Quality.h"
#include "SourceCode.h"
#include "URI.h"
#include "index/Index.h"
#include "index/Symbol.h"
#include "index/SymbolOrigin.h"
#include "support/Logger.h"
#include "support/Markup.h"
#include "support/Threading.h"
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/ExternalPreprocessorSource.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ScopedPrinter.h"
#include <algorithm>
#include <iterator>
#include <limits>

// We log detailed candidate here if you run with -debug-only=codecomplete.
#define DEBUG_TYPE "CodeComplete"

namespace clang {
namespace clangd {
namespace {

CompletionItemKind toCompletionItemKind(index::SymbolKind Kind) {
  using SK = index::SymbolKind;
  switch (Kind) {
  case SK::Unknown:
    return CompletionItemKind::Missing;
  case SK::Module:
  case SK::Namespace:
  case SK::NamespaceAlias:
    return CompletionItemKind::Module;
  case SK::Macro:
    return CompletionItemKind::Text;
  case SK::Enum:
    return CompletionItemKind::Enum;
  case SK::Struct:
    return CompletionItemKind::Struct;
  case SK::Class:
  case SK::Protocol:
  case SK::Extension:
  case SK::Union:
    return CompletionItemKind::Class;
  case SK::TypeAlias:
    // We use the same kind as the VSCode C++ extension.
    // FIXME: pick a better option when we have one.
    return CompletionItemKind::Interface;
  case SK::Using:
    return CompletionItemKind::Reference;
  case SK::Function:
  case SK::ConversionFunction:
    return CompletionItemKind::Function;
  case SK::Variable:
  case SK::Parameter:
  case SK::NonTypeTemplateParm:
    return CompletionItemKind::Variable;
  case SK::Field:
    return CompletionItemKind::Field;
  case SK::EnumConstant:
    return CompletionItemKind::EnumMember;
  case SK::InstanceMethod:
  case SK::ClassMethod:
  case SK::StaticMethod:
  case SK::Destructor:
    return CompletionItemKind::Method;
  case SK::InstanceProperty:
  case SK::ClassProperty:
  case SK::StaticProperty:
    return CompletionItemKind::Property;
  case SK::Constructor:
    return CompletionItemKind::Constructor;
  case SK::TemplateTypeParm:
  case SK::TemplateTemplateParm:
    return CompletionItemKind::TypeParameter;
  case SK::Concept:
    return CompletionItemKind::Interface;
  }
  llvm_unreachable("Unhandled clang::index::SymbolKind.");
}

CompletionItemKind
toCompletionItemKind(CodeCompletionResult::ResultKind ResKind,
                     const NamedDecl *Decl,
                     CodeCompletionContext::Kind CtxKind) {
  if (Decl)
    return toCompletionItemKind(index::getSymbolInfo(Decl).Kind);
  if (CtxKind == CodeCompletionContext::CCC_IncludedFile)
    return CompletionItemKind::File;
  switch (ResKind) {
  case CodeCompletionResult::RK_Declaration:
    llvm_unreachable("RK_Declaration without Decl");
  case CodeCompletionResult::RK_Keyword:
    return CompletionItemKind::Keyword;
  case CodeCompletionResult::RK_Macro:
    return CompletionItemKind::Text; // unfortunately, there's no 'Macro'
                                     // completion items in LSP.
  case CodeCompletionResult::RK_Pattern:
    return CompletionItemKind::Snippet;
  }
  llvm_unreachable("Unhandled CodeCompletionResult::ResultKind.");
}

// FIXME: find a home for this (that can depend on both markup and Protocol).
MarkupContent renderDoc(const markup::Document &Doc, MarkupKind Kind) {
  MarkupContent Result;
  Result.kind = Kind;
  switch (Kind) {
  case MarkupKind::PlainText:
    Result.value.append(Doc.asPlainText());
    break;
  case MarkupKind::Markdown:
    Result.value.append(Doc.asMarkdown());
    break;
  }
  return Result;
}

// Identifier code completion result.
struct RawIdentifier {
  llvm::StringRef Name;
  unsigned References; // # of usages in file.
};

/// A code completion result, in clang-native form.
/// It may be promoted to a CompletionItem if it's among the top-ranked results.
struct CompletionCandidate {
  llvm::StringRef Name; // Used for filtering and sorting.
  // We may have a result from Sema, from the index, or both.
  const CodeCompletionResult *SemaResult = nullptr;
  const Symbol *IndexResult = nullptr;
  const RawIdentifier *IdentifierResult = nullptr;
  llvm::SmallVector<llvm::StringRef, 1> RankedIncludeHeaders;

  // Returns a token identifying the overload set this is part of.
  // 0 indicates it's not part of any overload set.
  size_t overloadSet(const CodeCompleteOptions &Opts, llvm::StringRef FileName,
                     IncludeInserter *Inserter) const {
    if (!Opts.BundleOverloads.value_or(false))
      return 0;

    // Depending on the index implementation, we can see different header
    // strings (literal or URI) mapping to the same file. We still want to
    // bundle those, so we must resolve the header to be included here.
    std::string HeaderForHash;
    if (Inserter) {
      if (auto Header = headerToInsertIfAllowed(Opts)) {
        if (auto HeaderFile = toHeaderFile(*Header, FileName)) {
          if (auto Spelled =
                  Inserter->calculateIncludePath(*HeaderFile, FileName))
            HeaderForHash = *Spelled;
        } else {
          vlog("Code completion header path manipulation failed {0}",
               HeaderFile.takeError());
        }
      }
    }

    llvm::SmallString<256> Scratch;
    if (IndexResult) {
      switch (IndexResult->SymInfo.Kind) {
      case index::SymbolKind::ClassMethod:
      case index::SymbolKind::InstanceMethod:
      case index::SymbolKind::StaticMethod:
#ifndef NDEBUG
        llvm_unreachable("Don't expect members from index in code completion");
#else
        LLVM_FALLTHROUGH;
#endif
      case index::SymbolKind::Function:
        // We can't group overloads together that need different #includes.
        // This could break #include insertion.
        return llvm::hash_combine(
            (IndexResult->Scope + IndexResult->Name).toStringRef(Scratch),
            HeaderForHash);
      default:
        return 0;
      }
    }
    if (SemaResult) {
      // We need to make sure we're consistent with the IndexResult case!
      const NamedDecl *D = SemaResult->Declaration;
      if (!D || !D->isFunctionOrFunctionTemplate())
        return 0;
      {
        llvm::raw_svector_ostream OS(Scratch);
        D->printQualifiedName(OS);
      }
      return llvm::hash_combine(Scratch, HeaderForHash);
    }
    assert(IdentifierResult);
    return 0;
  }

  // The best header to include if include insertion is allowed.
  llvm::Optional<llvm::StringRef>
  headerToInsertIfAllowed(const CodeCompleteOptions &Opts) const {
    if (Opts.InsertIncludes == CodeCompleteOptions::NeverInsert ||
        RankedIncludeHeaders.empty())
      return None;
    if (SemaResult && SemaResult->Declaration) {
      // Avoid inserting new #include if the declaration is found in the current
      // file e.g. the symbol is forward declared.
      auto &SM = SemaResult->Declaration->getASTContext().getSourceManager();
      for (const Decl *RD : SemaResult->Declaration->redecls())
        if (SM.isInMainFile(SM.getExpansionLoc(RD->getBeginLoc())))
          return None;
    }
    return RankedIncludeHeaders[0];
  }

  using Bundle = llvm::SmallVector<CompletionCandidate, 4>;
};
using ScoredBundle =
    std::pair<CompletionCandidate::Bundle, CodeCompletion::Scores>;
struct ScoredBundleGreater {
  bool operator()(const ScoredBundle &L, const ScoredBundle &R) {
    if (L.second.Total != R.second.Total)
      return L.second.Total > R.second.Total;
    return L.first.front().Name <
           R.first.front().Name; // Earlier name is better.
  }
};

// Assembles a code completion out of a bundle of >=1 completion candidates.
// Many of the expensive strings are only computed at this point, once we know
// the candidate bundle is going to be returned.
//
// Many fields are the same for all candidates in a bundle (e.g. name), and are
// computed from the first candidate, in the constructor.
// Others vary per candidate, so add() must be called for remaining candidates.
struct CodeCompletionBuilder {
  CodeCompletionBuilder(ASTContext *ASTCtx, const CompletionCandidate &C,
                        CodeCompletionString *SemaCCS,
                        llvm::ArrayRef<std::string> QueryScopes,
                        const IncludeInserter &Includes,
                        llvm::StringRef FileName,
                        CodeCompletionContext::Kind ContextKind,
                        const CodeCompleteOptions &Opts,
                        bool IsUsingDeclaration, tok::TokenKind NextTokenKind)
      : ASTCtx(ASTCtx),
        EnableFunctionArgSnippets(Opts.EnableFunctionArgSnippets),
        IsUsingDeclaration(IsUsingDeclaration), NextTokenKind(NextTokenKind) {
    Completion.Deprecated = true; // cleared by any non-deprecated overload.
    add(C, SemaCCS);
    if (C.SemaResult) {
      assert(ASTCtx);
      Completion.Origin |= SymbolOrigin::AST;
      Completion.Name = std::string(llvm::StringRef(SemaCCS->getTypedText()));
      Completion.FilterText = SemaCCS->getAllTypedText();
      if (Completion.Scope.empty()) {
        if ((C.SemaResult->Kind == CodeCompletionResult::RK_Declaration) ||
            (C.SemaResult->Kind == CodeCompletionResult::RK_Pattern))
          if (const auto *D = C.SemaResult->getDeclaration())
            if (const auto *ND = dyn_cast<NamedDecl>(D))
              Completion.Scope = std::string(
                  splitQualifiedName(printQualifiedName(*ND)).first);
      }
      Completion.Kind = toCompletionItemKind(
          C.SemaResult->Kind, C.SemaResult->Declaration, ContextKind);
      // Sema could provide more info on whether the completion was a file or
      // folder.
      if (Completion.Kind == CompletionItemKind::File &&
          Completion.Name.back() == '/')
        Completion.Kind = CompletionItemKind::Folder;
      for (const auto &FixIt : C.SemaResult->FixIts) {
        Completion.FixIts.push_back(toTextEdit(
            FixIt, ASTCtx->getSourceManager(), ASTCtx->getLangOpts()));
      }
      llvm::sort(Completion.FixIts, [](const TextEdit &X, const TextEdit &Y) {
        return std::tie(X.range.start.line, X.range.start.character) <
               std::tie(Y.range.start.line, Y.range.start.character);
      });
    }
    if (C.IndexResult) {
      Completion.Origin |= C.IndexResult->Origin;
      if (Completion.Scope.empty())
        Completion.Scope = std::string(C.IndexResult->Scope);
      if (Completion.Kind == CompletionItemKind::Missing)
        Completion.Kind = toCompletionItemKind(C.IndexResult->SymInfo.Kind);
      if (Completion.Name.empty())
        Completion.Name = std::string(C.IndexResult->Name);
      if (Completion.FilterText.empty())
        Completion.FilterText = Completion.Name;
      // If the completion was visible to Sema, no qualifier is needed. This
      // avoids unneeded qualifiers in cases like with `using ns::X`.
      if (Completion.RequiredQualifier.empty() && !C.SemaResult) {
        llvm::StringRef ShortestQualifier = C.IndexResult->Scope;
        for (llvm::StringRef Scope : QueryScopes) {
          llvm::StringRef Qualifier = C.IndexResult->Scope;
          if (Qualifier.consume_front(Scope) &&
              Qualifier.size() < ShortestQualifier.size())
            ShortestQualifier = Qualifier;
        }
        Completion.RequiredQualifier = std::string(ShortestQualifier);
      }
    }
    if (C.IdentifierResult) {
      Completion.Origin |= SymbolOrigin::Identifier;
      Completion.Kind = CompletionItemKind::Text;
      Completion.Name = std::string(C.IdentifierResult->Name);
      Completion.FilterText = Completion.Name;
    }

    // Turn absolute path into a literal string that can be #included.
    auto Inserted = [&](llvm::StringRef Header)
        -> llvm::Expected<std::pair<std::string, bool>> {
      auto ResolvedDeclaring =
          URI::resolve(C.IndexResult->CanonicalDeclaration.FileURI, FileName);
      if (!ResolvedDeclaring)
        return ResolvedDeclaring.takeError();
      auto ResolvedInserted = toHeaderFile(Header, FileName);
      if (!ResolvedInserted)
        return ResolvedInserted.takeError();
      auto Spelled = Includes.calculateIncludePath(*ResolvedInserted, FileName);
      if (!Spelled)
        return error("Header not on include path");
      return std::make_pair(
          std::move(*Spelled),
          Includes.shouldInsertInclude(*ResolvedDeclaring, *ResolvedInserted));
    };
    bool ShouldInsert = C.headerToInsertIfAllowed(Opts).has_value();
    // Calculate include paths and edits for all possible headers.
    for (const auto &Inc : C.RankedIncludeHeaders) {
      if (auto ToInclude = Inserted(Inc)) {
        CodeCompletion::IncludeCandidate Include;
        Include.Header = ToInclude->first;
        if (ToInclude->second && ShouldInsert)
          Include.Insertion = Includes.insert(ToInclude->first);
        Completion.Includes.push_back(std::move(Include));
      } else
        log("Failed to generate include insertion edits for adding header "
            "(FileURI='{0}', IncludeHeader='{1}') into {2}: {3}",
            C.IndexResult->CanonicalDeclaration.FileURI, Inc, FileName,
            ToInclude.takeError());
    }
    // Prefer includes that do not need edits (i.e. already exist).
    std::stable_partition(Completion.Includes.begin(),
                          Completion.Includes.end(),
                          [](const CodeCompletion::IncludeCandidate &I) {
                            return !I.Insertion.has_value();
                          });
  }

  void add(const CompletionCandidate &C, CodeCompletionString *SemaCCS) {
    assert(bool(C.SemaResult) == bool(SemaCCS));
    Bundled.emplace_back();
    BundledEntry &S = Bundled.back();
    if (C.SemaResult) {
      bool IsPattern = C.SemaResult->Kind == CodeCompletionResult::RK_Pattern;
      getSignature(*SemaCCS, &S.Signature, &S.SnippetSuffix,
                   &Completion.RequiredQualifier, IsPattern);
      S.ReturnType = getReturnType(*SemaCCS);
    } else if (C.IndexResult) {
      S.Signature = std::string(C.IndexResult->Signature);
      S.SnippetSuffix = std::string(C.IndexResult->CompletionSnippetSuffix);
      S.ReturnType = std::string(C.IndexResult->ReturnType);
    }
    if (!Completion.Documentation) {
      auto SetDoc = [&](llvm::StringRef Doc) {
        if (!Doc.empty()) {
          Completion.Documentation.emplace();
          parseDocumentation(Doc, *Completion.Documentation);
        }
      };
      if (C.IndexResult) {
        SetDoc(C.IndexResult->Documentation);
      } else if (C.SemaResult) {
        const auto DocComment = getDocComment(*ASTCtx, *C.SemaResult,
                                              /*CommentsFromHeaders=*/false);
        SetDoc(formatDocumentation(*SemaCCS, DocComment));
      }
    }
    if (Completion.Deprecated) {
      if (C.SemaResult)
        Completion.Deprecated &=
            C.SemaResult->Availability == CXAvailability_Deprecated;
      if (C.IndexResult)
        Completion.Deprecated &=
            bool(C.IndexResult->Flags & Symbol::Deprecated);
    }
  }

  CodeCompletion build() {
    Completion.ReturnType = summarizeReturnType();
    Completion.Signature = summarizeSignature();
    Completion.SnippetSuffix = summarizeSnippet();
    Completion.BundleSize = Bundled.size();
    return std::move(Completion);
  }

private:
  struct BundledEntry {
    std::string SnippetSuffix;
    std::string Signature;
    std::string ReturnType;
  };

  // If all BundledEntries have the same value for a property, return it.
  template <std::string BundledEntry::*Member>
  const std::string *onlyValue() const {
    auto B = Bundled.begin(), E = Bundled.end();
    for (auto *I = B + 1; I != E; ++I)
      if (I->*Member != B->*Member)
        return nullptr;
    return &(B->*Member);
  }

  template <bool BundledEntry::*Member> const bool *onlyValue() const {
    auto B = Bundled.begin(), E = Bundled.end();
    for (auto *I = B + 1; I != E; ++I)
      if (I->*Member != B->*Member)
        return nullptr;
    return &(B->*Member);
  }

  std::string summarizeReturnType() const {
    if (auto *RT = onlyValue<&BundledEntry::ReturnType>())
      return *RT;
    return "";
  }

  std::string summarizeSnippet() const {
    if (IsUsingDeclaration)
      return "";
    auto *Snippet = onlyValue<&BundledEntry::SnippetSuffix>();
    if (!Snippet)
      // All bundles are function calls.
      // FIXME(ibiryukov): sometimes add template arguments to a snippet, e.g.
      // we need to complete 'forward<$1>($0)'.
      return "($0)";

    bool MayHaveArgList = Completion.Kind == CompletionItemKind::Function ||
                          Completion.Kind == CompletionItemKind::Method ||
                          Completion.Kind == CompletionItemKind::Constructor ||
                          Completion.Kind == CompletionItemKind::Text /*Macro*/;
    // If likely arg list already exists, don't add new parens & placeholders.
    //   Snippet: function(int x, int y)
    //   func^(1,2) -> function(1, 2)
    //             NOT function(int x, int y)(1, 2)
    if (MayHaveArgList) {
      // Check for a template argument list in the code.
      //   Snippet: function<class T>(int x)
      //   fu^<int>(1) -> function<int>(1)
      if (NextTokenKind == tok::less && Snippet->front() == '<')
        return "";
      // Potentially followed by regular argument list.
      if (NextTokenKind == tok::l_paren) {
        //   Snippet: function<class T>(int x)
        //   fu^(1,2) -> function<class T>(1, 2)
        if (Snippet->front() == '<') {
          // Find matching '>', handling nested brackets.
          int Balance = 0;
          size_t I = 0;
          do {
            if (Snippet->at(I) == '>')
              --Balance;
            else if (Snippet->at(I) == '<')
              ++Balance;
            ++I;
          } while (Balance > 0);
          return Snippet->substr(0, I);
        }
        return "";
      }
    }
    if (EnableFunctionArgSnippets)
      return *Snippet;

    // Replace argument snippets with a simplified pattern.
    if (Snippet->empty())
      return "";
    if (MayHaveArgList) {
      // Functions snippets can be of 2 types:
      // - containing only function arguments, e.g.
      //   foo(${1:int p1}, ${2:int p2});
      //   We transform this pattern to '($0)' or '()'.
      // - template arguments and function arguments, e.g.
      //   foo<${1:class}>(${2:int p1}).
      //   We transform this pattern to '<$1>()$0' or '<$0>()'.

      bool EmptyArgs = llvm::StringRef(*Snippet).endswith("()");
      if (Snippet->front() == '<')
        return EmptyArgs ? "<$1>()$0" : "<$1>($0)";
      if (Snippet->front() == '(')
        return EmptyArgs ? "()" : "($0)";
      return *Snippet; // Not an arg snippet?
    }
    // 'CompletionItemKind::Interface' matches template type aliases.
    if (Completion.Kind == CompletionItemKind::Interface ||
        Completion.Kind == CompletionItemKind::Class) {
      if (Snippet->front() != '<')
        return *Snippet; // Not an arg snippet?

      // Classes and template using aliases can only have template arguments,
      // e.g. Foo<${1:class}>.
      if (llvm::StringRef(*Snippet).endswith("<>"))
        return "<>"; // can happen with defaulted template arguments.
      return "<$0>";
    }
    return *Snippet;
  }

  std::string summarizeSignature() const {
    if (auto *Signature = onlyValue<&BundledEntry::Signature>())
      return *Signature;
    // All bundles are function calls.
    return "(…)";
  }

  // ASTCtx can be nullptr if not run with sema.
  ASTContext *ASTCtx;
  CodeCompletion Completion;
  llvm::SmallVector<BundledEntry, 1> Bundled;
  bool EnableFunctionArgSnippets;
  // No snippets will be generated for using declarations and when the function
  // arguments are already present.
  bool IsUsingDeclaration;
  tok::TokenKind NextTokenKind;
};

// Determine the symbol ID for a Sema code completion result, if possible.
SymbolID getSymbolID(const CodeCompletionResult &R, const SourceManager &SM) {
  switch (R.Kind) {
  case CodeCompletionResult::RK_Declaration:
  case CodeCompletionResult::RK_Pattern: {
    // Computing USR caches linkage, which may change after code completion.
    if (hasUnstableLinkage(R.Declaration))
      return {};
    return clang::clangd::getSymbolID(R.Declaration);
  }
  case CodeCompletionResult::RK_Macro:
    return clang::clangd::getSymbolID(R.Macro->getName(), R.MacroDefInfo, SM);
  case CodeCompletionResult::RK_Keyword:
    return {};
  }
  llvm_unreachable("unknown CodeCompletionResult kind");
}

// Scopes of the partial identifier we're trying to complete.
// It is used when we query the index for more completion results.
struct SpecifiedScope {
  // The scopes we should look in, determined by Sema.
  //
  // If the qualifier was fully resolved, we look for completions in these
  // scopes; if there is an unresolved part of the qualifier, it should be
  // resolved within these scopes.
  //
  // Examples of qualified completion:
  //
  //   "::vec"                                      => {""}
  //   "using namespace std; ::vec^"                => {"", "std::"}
  //   "namespace ns {using namespace std;} ns::^"  => {"ns::", "std::"}
  //   "std::vec^"                                  => {""}  // "std" unresolved
  //
  // Examples of unqualified completion:
  //
  //   "vec^"                                       => {""}
  //   "using namespace std; vec^"                  => {"", "std::"}
  //   "using namespace std; namespace ns { vec^ }" => {"ns::", "std::", ""}
  //
  // "" for global namespace, "ns::" for normal namespace.
  std::vector<std::string> AccessibleScopes;
  // The full scope qualifier as typed by the user (without the leading "::").
  // Set if the qualifier is not fully resolved by Sema.
  llvm::Optional<std::string> UnresolvedQualifier;

  // Construct scopes being queried in indexes. The results are deduplicated.
  // This method format the scopes to match the index request representation.
  std::vector<std::string> scopesForIndexQuery() {
    std::set<std::string> Results;
    for (llvm::StringRef AS : AccessibleScopes)
      Results.insert(
          (AS + (UnresolvedQualifier ? *UnresolvedQualifier : "")).str());
    return {Results.begin(), Results.end()};
  }
};

// Get all scopes that will be queried in indexes and whether symbols from
// any scope is allowed. The first scope in the list is the preferred scope
// (e.g. enclosing namespace).
std::pair<std::vector<std::string>, bool>
getQueryScopes(CodeCompletionContext &CCContext, const Sema &CCSema,
               const CompletionPrefix &HeuristicPrefix,
               const CodeCompleteOptions &Opts) {
  SpecifiedScope Scopes;
  for (auto *Context : CCContext.getVisitedContexts()) {
    if (isa<TranslationUnitDecl>(Context))
      Scopes.AccessibleScopes.push_back(""); // global namespace
    else if (isa<NamespaceDecl>(Context))
      Scopes.AccessibleScopes.push_back(printNamespaceScope(*Context));
  }

  const CXXScopeSpec *SemaSpecifier =
      CCContext.getCXXScopeSpecifier().value_or(nullptr);
  // Case 1: unqualified completion.
  if (!SemaSpecifier) {
    // Case 2 (exception): sema saw no qualifier, but there appears to be one!
    // This can happen e.g. in incomplete macro expansions. Use heuristics.
    if (!HeuristicPrefix.Qualifier.empty()) {
      vlog("Sema said no scope specifier, but we saw {0} in the source code",
           HeuristicPrefix.Qualifier);
      StringRef SpelledSpecifier = HeuristicPrefix.Qualifier;
      if (SpelledSpecifier.consume_front("::"))
        Scopes.AccessibleScopes = {""};
      Scopes.UnresolvedQualifier = std::string(SpelledSpecifier);
      return {Scopes.scopesForIndexQuery(), false};
    }
    // The enclosing namespace must be first, it gets a quality boost.
    std::vector<std::string> EnclosingAtFront;
    std::string EnclosingScope = printNamespaceScope(*CCSema.CurContext);
    EnclosingAtFront.push_back(EnclosingScope);
    for (auto &S : Scopes.scopesForIndexQuery()) {
      if (EnclosingScope != S)
        EnclosingAtFront.push_back(std::move(S));
    }
    // Allow AllScopes completion as there is no explicit scope qualifier.
    return {EnclosingAtFront, Opts.AllScopes};
  }
  // Case 3: sema saw and resolved a scope qualifier.
  if (SemaSpecifier && SemaSpecifier->isValid())
    return {Scopes.scopesForIndexQuery(), false};

  // Case 4: There was a qualifier, and Sema didn't resolve it.
  Scopes.AccessibleScopes.push_back(""); // Make sure global scope is included.
  llvm::StringRef SpelledSpecifier = Lexer::getSourceText(
      CharSourceRange::getCharRange(SemaSpecifier->getRange()),
      CCSema.SourceMgr, clang::LangOptions());
  if (SpelledSpecifier.consume_front("::"))
    Scopes.AccessibleScopes = {""};
  Scopes.UnresolvedQualifier = std::string(SpelledSpecifier);
  // Sema excludes the trailing "::".
  if (!Scopes.UnresolvedQualifier->empty())
    *Scopes.UnresolvedQualifier += "::";

  return {Scopes.scopesForIndexQuery(), false};
}

// Should we perform index-based completion in a context of the specified kind?
// FIXME: consider allowing completion, but restricting the result types.
bool contextAllowsIndex(enum CodeCompletionContext::Kind K) {
  switch (K) {
  case CodeCompletionContext::CCC_TopLevel:
  case CodeCompletionContext::CCC_ObjCInterface:
  case CodeCompletionContext::CCC_ObjCImplementation:
  case CodeCompletionContext::CCC_ObjCIvarList:
  case CodeCompletionContext::CCC_ClassStructUnion:
  case CodeCompletionContext::CCC_Statement:
  case CodeCompletionContext::CCC_Expression:
  case CodeCompletionContext::CCC_ObjCMessageReceiver:
  case CodeCompletionContext::CCC_EnumTag:
  case CodeCompletionContext::CCC_UnionTag:
  case CodeCompletionContext::CCC_ClassOrStructTag:
  case CodeCompletionContext::CCC_ObjCProtocolName:
  case CodeCompletionContext::CCC_Namespace:
  case CodeCompletionContext::CCC_Type:
  case CodeCompletionContext::CCC_ParenthesizedExpression:
  case CodeCompletionContext::CCC_ObjCInterfaceName:
  case CodeCompletionContext::CCC_ObjCCategoryName:
  case CodeCompletionContext::CCC_Symbol:
  case CodeCompletionContext::CCC_SymbolOrNewName:
    return true;
  case CodeCompletionContext::CCC_OtherWithMacros:
  case CodeCompletionContext::CCC_DotMemberAccess:
  case CodeCompletionContext::CCC_ArrowMemberAccess:
  case CodeCompletionContext::CCC_ObjCPropertyAccess:
  case CodeCompletionContext::CCC_MacroName:
  case CodeCompletionContext::CCC_MacroNameUse:
  case CodeCompletionContext::CCC_PreprocessorExpression:
  case CodeCompletionContext::CCC_PreprocessorDirective:
  case CodeCompletionContext::CCC_SelectorName:
  case CodeCompletionContext::CCC_TypeQualifiers:
  case CodeCompletionContext::CCC_ObjCInstanceMessage:
  case CodeCompletionContext::CCC_ObjCClassMessage:
  case CodeCompletionContext::CCC_IncludedFile:
  case CodeCompletionContext::CCC_Attribute:
  // FIXME: Provide identifier based completions for the following contexts:
  case CodeCompletionContext::CCC_Other: // Be conservative.
  case CodeCompletionContext::CCC_NaturalLanguage:
  case CodeCompletionContext::CCC_Recovery:
  case CodeCompletionContext::CCC_NewName:
    return false;
  }
  llvm_unreachable("unknown code completion context");
}

static bool isInjectedClass(const NamedDecl &D) {
  if (auto *R = dyn_cast_or_null<RecordDecl>(&D))
    if (R->isInjectedClassName())
      return true;
  return false;
}

// Some member calls are excluded because they're so rarely useful.
static bool isExcludedMember(const NamedDecl &D) {
  // Destructor completion is rarely useful, and works inconsistently.
  // (s.^ completes ~string, but s.~st^ is an error).
  if (D.getKind() == Decl::CXXDestructor)
    return true;
  // Injected name may be useful for A::foo(), but who writes A::A::foo()?
  if (isInjectedClass(D))
    return true;
  // Explicit calls to operators are also rare.
  auto NameKind = D.getDeclName().getNameKind();
  if (NameKind == DeclarationName::CXXOperatorName ||
      NameKind == DeclarationName::CXXLiteralOperatorName ||
      NameKind == DeclarationName::CXXConversionFunctionName)
    return true;
  return false;
}

// The CompletionRecorder captures Sema code-complete output, including context.
// It filters out ignored results (but doesn't apply fuzzy-filtering yet).
// It doesn't do scoring or conversion to CompletionItem yet, as we want to
// merge with index results first.
// Generally the fields and methods of this object should only be used from
// within the callback.
struct CompletionRecorder : public CodeCompleteConsumer {
  CompletionRecorder(const CodeCompleteOptions &Opts,
                     llvm::unique_function<void()> ResultsCallback)
      : CodeCompleteConsumer(Opts.getClangCompleteOpts()),
        CCContext(CodeCompletionContext::CCC_Other), Opts(Opts),
        CCAllocator(std::make_shared<GlobalCodeCompletionAllocator>()),
        CCTUInfo(CCAllocator), ResultsCallback(std::move(ResultsCallback)) {
    assert(this->ResultsCallback);
  }

  std::vector<CodeCompletionResult> Results;
  CodeCompletionContext CCContext;
  Sema *CCSema = nullptr; // Sema that created the results.
  // FIXME: Sema is scary. Can we store ASTContext and Preprocessor, instead?

  void ProcessCodeCompleteResults(class Sema &S, CodeCompletionContext Context,
                                  CodeCompletionResult *InResults,
                                  unsigned NumResults) final {
    // Results from recovery mode are generally useless, and the callback after
    // recovery (if any) is usually more interesting. To make sure we handle the
    // future callback from sema, we just ignore all callbacks in recovery mode,
    // as taking only results from recovery mode results in poor completion
    // results.
    // FIXME: in case there is no future sema completion callback after the
    // recovery mode, we might still want to provide some results (e.g. trivial
    // identifier-based completion).
    if (Context.getKind() == CodeCompletionContext::CCC_Recovery) {
      log("Code complete: Ignoring sema code complete callback with Recovery "
          "context.");
      return;
    }
    // If a callback is called without any sema result and the context does not
    // support index-based completion, we simply skip it to give way to
    // potential future callbacks with results.
    if (NumResults == 0 && !contextAllowsIndex(Context.getKind()))
      return;
    if (CCSema) {
      log("Multiple code complete callbacks (parser backtracked?). "
          "Dropping results from context {0}, keeping results from {1}.",
          getCompletionKindString(Context.getKind()),
          getCompletionKindString(this->CCContext.getKind()));
      return;
    }
    // Record the completion context.
    CCSema = &S;
    CCContext = Context;

    // Retain the results we might want.
    for (unsigned I = 0; I < NumResults; ++I) {
      auto &Result = InResults[I];
      // Class members that are shadowed by subclasses are usually noise.
      if (Result.Hidden && Result.Declaration &&
          Result.Declaration->isCXXClassMember())
        continue;
      if (!Opts.IncludeIneligibleResults &&
          (Result.Availability == CXAvailability_NotAvailable ||
           Result.Availability == CXAvailability_NotAccessible))
        continue;
      if (Result.Declaration &&
          !Context.getBaseType().isNull() // is this a member-access context?
          && isExcludedMember(*Result.Declaration))
        continue;
      // Skip injected class name when no class scope is not explicitly set.
      // E.g. show injected A::A in `using A::A^` but not in "A^".
      if (Result.Declaration && !Context.getCXXScopeSpecifier() &&
          isInjectedClass(*Result.Declaration))
        continue;
      // We choose to never append '::' to completion results in clangd.
      Result.StartsNestedNameSpecifier = false;
      Results.push_back(Result);
    }
    ResultsCallback();
  }

  CodeCompletionAllocator &getAllocator() override { return *CCAllocator; }
  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }

  // Returns the filtering/sorting name for Result, which must be from Results.
  // Returned string is owned by this recorder (or the AST).
  llvm::StringRef getName(const CodeCompletionResult &Result) {
    switch (Result.Kind) {
    case CodeCompletionResult::RK_Declaration:
      if (auto *ID = Result.Declaration->getIdentifier())
        return ID->getName();
      break;
    case CodeCompletionResult::RK_Keyword:
      return Result.Keyword;
    case CodeCompletionResult::RK_Macro:
      return Result.Macro->getName();
    case CodeCompletionResult::RK_Pattern:
      break;
    }
    auto *CCS = codeCompletionString(Result);
    const CodeCompletionString::Chunk *OnlyText = nullptr;
    for (auto &C : *CCS) {
      if (C.Kind != CodeCompletionString::CK_TypedText)
        continue;
      if (OnlyText)
        return CCAllocator->CopyString(CCS->getAllTypedText());
      OnlyText = &C;
    }
    return OnlyText ? OnlyText->Text : llvm::StringRef();
  }

  // Build a CodeCompletion string for R, which must be from Results.
  // The CCS will be owned by this recorder.
  CodeCompletionString *codeCompletionString(const CodeCompletionResult &R) {
    // CodeCompletionResult doesn't seem to be const-correct. We own it, anyway.
    return const_cast<CodeCompletionResult &>(R).CreateCodeCompletionString(
        *CCSema, CCContext, *CCAllocator, CCTUInfo,
        /*IncludeBriefComments=*/false);
  }

private:
  CodeCompleteOptions Opts;
  std::shared_ptr<GlobalCodeCompletionAllocator> CCAllocator;
  CodeCompletionTUInfo CCTUInfo;
  llvm::unique_function<void()> ResultsCallback;
};

struct ScoredSignature {
  // When not null, requires documentation to be requested from the index with
  // this ID.
  SymbolID IDForDoc;
  SignatureInformation Signature;
  SignatureQualitySignals Quality;
};

// Returns the index of the parameter matching argument number "Arg.
// This is usually just "Arg", except for variadic functions/templates, where
// "Arg" might be higher than the number of parameters. When that happens, we
// assume the last parameter is variadic and assume all further args are
// part of it.
int paramIndexForArg(const CodeCompleteConsumer::OverloadCandidate &Candidate,
                     int Arg) {
  int NumParams = Candidate.getNumParams();
  if (auto *T = Candidate.getFunctionType()) {
    if (auto *Proto = T->getAs<FunctionProtoType>()) {
      if (Proto->isVariadic())
        ++NumParams;
    }
  }
  return std::min(Arg, std::max(NumParams - 1, 0));
}

class SignatureHelpCollector final : public CodeCompleteConsumer {
public:
  SignatureHelpCollector(const clang::CodeCompleteOptions &CodeCompleteOpts,
                         MarkupKind DocumentationFormat,
                         const SymbolIndex *Index, SignatureHelp &SigHelp)
      : CodeCompleteConsumer(CodeCompleteOpts), SigHelp(SigHelp),
        Allocator(std::make_shared<clang::GlobalCodeCompletionAllocator>()),
        CCTUInfo(Allocator), Index(Index),
        DocumentationFormat(DocumentationFormat) {}

  void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                 OverloadCandidate *Candidates,
                                 unsigned NumCandidates,
                                 SourceLocation OpenParLoc,
                                 bool Braced) override {
    assert(!OpenParLoc.isInvalid());
    SourceManager &SrcMgr = S.getSourceManager();
    OpenParLoc = SrcMgr.getFileLoc(OpenParLoc);
    if (SrcMgr.isInMainFile(OpenParLoc))
      SigHelp.argListStart = sourceLocToPosition(SrcMgr, OpenParLoc);
    else
      elog("Location oustide main file in signature help: {0}",
           OpenParLoc.printToString(SrcMgr));

    std::vector<ScoredSignature> ScoredSignatures;
    SigHelp.signatures.reserve(NumCandidates);
    ScoredSignatures.reserve(NumCandidates);
    // FIXME(rwols): How can we determine the "active overload candidate"?
    // Right now the overloaded candidates seem to be provided in a "best fit"
    // order, so I'm not too worried about this.
    SigHelp.activeSignature = 0;
    assert(CurrentArg <= (unsigned)std::numeric_limits<int>::max() &&
           "too many arguments");

    SigHelp.activeParameter = static_cast<int>(CurrentArg);

    for (unsigned I = 0; I < NumCandidates; ++I) {
      OverloadCandidate Candidate = Candidates[I];
      // We want to avoid showing instantiated signatures, because they may be
      // long in some cases (e.g. when 'T' is substituted with 'std::string', we
      // would get 'std::basic_string<char>').
      if (auto *Func = Candidate.getFunction()) {
        if (auto *Pattern = Func->getTemplateInstantiationPattern())
          Candidate = OverloadCandidate(Pattern);
      }
      if (static_cast<int>(I) == SigHelp.activeSignature) {
        // The activeParameter in LSP relates to the activeSignature. There is
        // another, per-signature field, but we currently do not use it and not
        // all clients might support it.
        // FIXME: Add support for per-signature activeParameter field.
        SigHelp.activeParameter =
            paramIndexForArg(Candidate, SigHelp.activeParameter);
      }

      const auto *CCS = Candidate.CreateSignatureString(
          CurrentArg, S, *Allocator, CCTUInfo,
          /*IncludeBriefComments=*/true, Braced);
      assert(CCS && "Expected the CodeCompletionString to be non-null");
      ScoredSignatures.push_back(processOverloadCandidate(
          Candidate, *CCS,
          Candidate.getFunction()
              ? getDeclComment(S.getASTContext(), *Candidate.getFunction())
              : ""));
    }

    // Sema does not load the docs from the preamble, so we need to fetch extra
    // docs from the index instead.
    llvm::DenseMap<SymbolID, std::string> FetchedDocs;
    if (Index) {
      LookupRequest IndexRequest;
      for (const auto &S : ScoredSignatures) {
        if (!S.IDForDoc)
          continue;
        IndexRequest.IDs.insert(S.IDForDoc);
      }
      Index->lookup(IndexRequest, [&](const Symbol &S) {
        if (!S.Documentation.empty())
          FetchedDocs[S.ID] = std::string(S.Documentation);
      });
      vlog("SigHelp: requested docs for {0} symbols from the index, got {1} "
           "symbols with non-empty docs in the response",
           IndexRequest.IDs.size(), FetchedDocs.size());
    }

    llvm::sort(ScoredSignatures, [](const ScoredSignature &L,
                                    const ScoredSignature &R) {
      // Ordering follows:
      // - Less number of parameters is better.
      // - Aggregate > Function > FunctionType > FunctionTemplate
      // - High score is better.
      // - Shorter signature is better.
      // - Alphabetically smaller is better.
      if (L.Quality.NumberOfParameters != R.Quality.NumberOfParameters)
        return L.Quality.NumberOfParameters < R.Quality.NumberOfParameters;
      if (L.Quality.NumberOfOptionalParameters !=
          R.Quality.NumberOfOptionalParameters)
        return L.Quality.NumberOfOptionalParameters <
               R.Quality.NumberOfOptionalParameters;
      if (L.Quality.Kind != R.Quality.Kind) {
        using OC = CodeCompleteConsumer::OverloadCandidate;
        auto KindPriority = [&](OC::CandidateKind K) {
          switch (K) {
          case OC::CK_Aggregate:
            return 0;
          case OC::CK_Function:
            return 1;
          case OC::CK_FunctionType:
            return 2;
          case OC::CK_FunctionProtoTypeLoc:
            return 3;
          case OC::CK_FunctionTemplate:
            return 4;
          case OC::CK_Template:
            return 5;
          }
          llvm_unreachable("Unknown overload candidate type.");
        };
        return KindPriority(L.Quality.Kind) < KindPriority(R.Quality.Kind);
      }
      if (L.Signature.label.size() != R.Signature.label.size())
        return L.Signature.label.size() < R.Signature.label.size();
      return L.Signature.label < R.Signature.label;
    });

    for (auto &SS : ScoredSignatures) {
      auto IndexDocIt =
          SS.IDForDoc ? FetchedDocs.find(SS.IDForDoc) : FetchedDocs.end();
      if (IndexDocIt != FetchedDocs.end()) {
        markup::Document SignatureComment;
        parseDocumentation(IndexDocIt->second, SignatureComment);
        SS.Signature.documentation =
            renderDoc(SignatureComment, DocumentationFormat);
      }

      SigHelp.signatures.push_back(std::move(SS.Signature));
    }
  }

  GlobalCodeCompletionAllocator &getAllocator() override { return *Allocator; }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }

private:
  void processParameterChunk(llvm::StringRef ChunkText,
                             SignatureInformation &Signature) const {
    // (!) this is O(n), should still be fast compared to building ASTs.
    unsigned ParamStartOffset = lspLength(Signature.label);
    unsigned ParamEndOffset = ParamStartOffset + lspLength(ChunkText);
    // A piece of text that describes the parameter that corresponds to
    // the code-completion location within a function call, message send,
    // macro invocation, etc.
    Signature.label += ChunkText;
    ParameterInformation Info;
    Info.labelOffsets.emplace(ParamStartOffset, ParamEndOffset);
    // FIXME: only set 'labelOffsets' when all clients migrate out of it.
    Info.labelString = std::string(ChunkText);

    Signature.parameters.push_back(std::move(Info));
  }

  void processOptionalChunk(const CodeCompletionString &CCS,
                            SignatureInformation &Signature,
                            SignatureQualitySignals &Signal) const {
    for (const auto &Chunk : CCS) {
      switch (Chunk.Kind) {
      case CodeCompletionString::CK_Optional:
        assert(Chunk.Optional &&
               "Expected the optional code completion string to be non-null.");
        processOptionalChunk(*Chunk.Optional, Signature, Signal);
        break;
      case CodeCompletionString::CK_VerticalSpace:
        break;
      case CodeCompletionString::CK_CurrentParameter:
      case CodeCompletionString::CK_Placeholder:
        processParameterChunk(Chunk.Text, Signature);
        Signal.NumberOfOptionalParameters++;
        break;
      default:
        Signature.label += Chunk.Text;
        break;
      }
    }
  }

  // FIXME(ioeric): consider moving CodeCompletionString logic here to
  // CompletionString.h.
  ScoredSignature processOverloadCandidate(const OverloadCandidate &Candidate,
                                           const CodeCompletionString &CCS,
                                           llvm::StringRef DocComment) const {
    SignatureInformation Signature;
    SignatureQualitySignals Signal;
    const char *ReturnType = nullptr;

    markup::Document OverloadComment;
    parseDocumentation(formatDocumentation(CCS, DocComment), OverloadComment);
    Signature.documentation = renderDoc(OverloadComment, DocumentationFormat);
    Signal.Kind = Candidate.getKind();

    for (const auto &Chunk : CCS) {
      switch (Chunk.Kind) {
      case CodeCompletionString::CK_ResultType:
        // A piece of text that describes the type of an entity or,
        // for functions and methods, the return type.
        assert(!ReturnType && "Unexpected CK_ResultType");
        ReturnType = Chunk.Text;
        break;
      case CodeCompletionString::CK_CurrentParameter:
      case CodeCompletionString::CK_Placeholder:
        processParameterChunk(Chunk.Text, Signature);
        Signal.NumberOfParameters++;
        break;
      case CodeCompletionString::CK_Optional: {
        // The rest of the parameters are defaulted/optional.
        assert(Chunk.Optional &&
               "Expected the optional code completion string to be non-null.");
        processOptionalChunk(*Chunk.Optional, Signature, Signal);
        break;
      }
      case CodeCompletionString::CK_VerticalSpace:
        break;
      default:
        Signature.label += Chunk.Text;
        break;
      }
    }
    if (ReturnType) {
      Signature.label += " -> ";
      Signature.label += ReturnType;
    }
    dlog("Signal for {0}: {1}", Signature, Signal);
    ScoredSignature Result;
    Result.Signature = std::move(Signature);
    Result.Quality = Signal;
    const FunctionDecl *Func = Candidate.getFunction();
    if (Func && Result.Signature.documentation.value.empty()) {
      // Computing USR caches linkage, which may change after code completion.
      if (!hasUnstableLinkage(Func))
        Result.IDForDoc = clangd::getSymbolID(Func);
    }
    return Result;
  }

  SignatureHelp &SigHelp;
  std::shared_ptr<clang::GlobalCodeCompletionAllocator> Allocator;
  CodeCompletionTUInfo CCTUInfo;
  const SymbolIndex *Index;
  MarkupKind DocumentationFormat;
}; // SignatureHelpCollector

// Used only for completion of C-style comments in function call (i.e.
// /*foo=*/7). Similar to SignatureHelpCollector, but needs to do less work.
class ParamNameCollector final : public CodeCompleteConsumer {
public:
  ParamNameCollector(const clang::CodeCompleteOptions &CodeCompleteOpts,
                     std::set<std::string> &ParamNames)
      : CodeCompleteConsumer(CodeCompleteOpts),
        Allocator(std::make_shared<clang::GlobalCodeCompletionAllocator>()),
        CCTUInfo(Allocator), ParamNames(ParamNames) {}

  void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                 OverloadCandidate *Candidates,
                                 unsigned NumCandidates,
                                 SourceLocation OpenParLoc,
                                 bool Braced) override {
    assert(CurrentArg <= (unsigned)std::numeric_limits<int>::max() &&
           "too many arguments");

    for (unsigned I = 0; I < NumCandidates; ++I) {
      if (const NamedDecl *ND = Candidates[I].getParamDecl(CurrentArg))
        if (const auto *II = ND->getIdentifier())
          ParamNames.emplace(II->getName());
    }
  }

private:
  GlobalCodeCompletionAllocator &getAllocator() override { return *Allocator; }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }

  std::shared_ptr<clang::GlobalCodeCompletionAllocator> Allocator;
  CodeCompletionTUInfo CCTUInfo;
  std::set<std::string> &ParamNames;
};

struct SemaCompleteInput {
  PathRef FileName;
  size_t Offset;
  const PreambleData &Preamble;
  const llvm::Optional<PreamblePatch> Patch;
  const ParseInputs &ParseInput;
};

void loadMainFilePreambleMacros(const Preprocessor &PP,
                                const PreambleData &Preamble) {
  // The ExternalPreprocessorSource has our macros, if we know where to look.
  // We can read all the macros using PreambleMacros->ReadDefinedMacros(),
  // but this includes transitively included files, so may deserialize a lot.
  ExternalPreprocessorSource *PreambleMacros = PP.getExternalSource();
  // As we have the names of the macros, we can look up their IdentifierInfo
  // and then use this to load just the macros we want.
  const auto &ITable = PP.getIdentifierTable();
  IdentifierInfoLookup *PreambleIdentifiers =
      ITable.getExternalIdentifierLookup();

  if (!PreambleIdentifiers || !PreambleMacros)
    return;
  for (const auto &MacroName : Preamble.Macros.Names) {
    if (ITable.find(MacroName.getKey()) != ITable.end())
      continue;
    if (auto *II = PreambleIdentifiers->get(MacroName.getKey()))
      if (II->isOutOfDate())
        PreambleMacros->updateOutOfDateIdentifier(*II);
  }
}

// Invokes Sema code completion on a file.
// If \p Includes is set, it will be updated based on the compiler invocation.
bool semaCodeComplete(std::unique_ptr<CodeCompleteConsumer> Consumer,
                      const clang::CodeCompleteOptions &Options,
                      const SemaCompleteInput &Input,
                      IncludeStructure *Includes = nullptr) {
  trace::Span Tracer("Sema completion");

  IgnoreDiagnostics IgnoreDiags;
  auto CI = buildCompilerInvocation(Input.ParseInput, IgnoreDiags);
  if (!CI) {
    elog("Couldn't create CompilerInvocation");
    return false;
  }
  auto &FrontendOpts = CI->getFrontendOpts();
  FrontendOpts.SkipFunctionBodies = true;
  // Disable typo correction in Sema.
  CI->getLangOpts()->SpellChecking = false;
  // Code completion won't trigger in delayed template bodies.
  // This is on-by-default in windows to allow parsing SDK headers; we're only
  // disabling it for the main-file (not preamble).
  CI->getLangOpts()->DelayedTemplateParsing = false;
  // Setup code completion.
  FrontendOpts.CodeCompleteOpts = Options;
  FrontendOpts.CodeCompletionAt.FileName = std::string(Input.FileName);
  std::tie(FrontendOpts.CodeCompletionAt.Line,
           FrontendOpts.CodeCompletionAt.Column) =
      offsetToClangLineColumn(Input.ParseInput.Contents, Input.Offset);

  std::unique_ptr<llvm::MemoryBuffer> ContentsBuffer =
      llvm::MemoryBuffer::getMemBuffer(Input.ParseInput.Contents,
                                       Input.FileName);
  // The diagnostic options must be set before creating a CompilerInstance.
  CI->getDiagnosticOpts().IgnoreWarnings = true;
  // We reuse the preamble whether it's valid or not. This is a
  // correctness/performance tradeoff: building without a preamble is slow, and
  // completion is latency-sensitive.
  // However, if we're completing *inside* the preamble section of the draft,
  // overriding the preamble will break sema completion. Fortunately we can just
  // skip all includes in this case; these completions are really simple.
  PreambleBounds PreambleRegion =
      ComputePreambleBounds(*CI->getLangOpts(), *ContentsBuffer, 0);
  bool CompletingInPreamble = Input.Offset < PreambleRegion.Size ||
                              (!PreambleRegion.PreambleEndsAtStartOfLine &&
                               Input.Offset == PreambleRegion.Size);
  if (Input.Patch)
    Input.Patch->apply(*CI);
  // NOTE: we must call BeginSourceFile after prepareCompilerInstance. Otherwise
  // the remapped buffers do not get freed.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS =
      Input.ParseInput.TFS->view(Input.ParseInput.CompileCommand.Directory);
  if (Input.Preamble.StatCache)
    VFS = Input.Preamble.StatCache->getConsumingFS(std::move(VFS));
  auto Clang = prepareCompilerInstance(
      std::move(CI), !CompletingInPreamble ? &Input.Preamble.Preamble : nullptr,
      std::move(ContentsBuffer), std::move(VFS), IgnoreDiags);
  Clang->getPreprocessorOpts().SingleFileParseMode = CompletingInPreamble;
  Clang->setCodeCompletionConsumer(Consumer.release());

  SyntaxOnlyAction Action;
  if (!Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0])) {
    log("BeginSourceFile() failed when running codeComplete for {0}",
        Input.FileName);
    return false;
  }
  // Macros can be defined within the preamble region of the main file.
  // They don't fall nicely into our index/Sema dichotomy:
  //  - they're not indexed for completion (they're not available across files)
  //  - but Sema code complete won't see them: as part of the preamble, they're
  //    deserialized only when mentioned.
  // Force them to be deserialized so SemaCodeComplete sees them.
  loadMainFilePreambleMacros(Clang->getPreprocessor(), Input.Preamble);
  if (Includes)
    Includes->collect(*Clang);
  if (llvm::Error Err = Action.Execute()) {
    log("Execute() failed when running codeComplete for {0}: {1}",
        Input.FileName, toString(std::move(Err)));
    return false;
  }
  Action.EndSourceFile();

  return true;
}

// Should we allow index completions in the specified context?
bool allowIndex(CodeCompletionContext &CC) {
  if (!contextAllowsIndex(CC.getKind()))
    return false;
  // We also avoid ClassName::bar (but allow namespace::bar).
  auto Scope = CC.getCXXScopeSpecifier();
  if (!Scope)
    return true;
  NestedNameSpecifier *NameSpec = (*Scope)->getScopeRep();
  if (!NameSpec)
    return true;
  // We only query the index when qualifier is a namespace.
  // If it's a class, we rely solely on sema completions.
  switch (NameSpec->getKind()) {
  case NestedNameSpecifier::Global:
  case NestedNameSpecifier::Namespace:
  case NestedNameSpecifier::NamespaceAlias:
    return true;
  case NestedNameSpecifier::Super:
  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate:
  // Unresolved inside a template.
  case NestedNameSpecifier::Identifier:
    return false;
  }
  llvm_unreachable("invalid NestedNameSpecifier kind");
}

std::future<SymbolSlab> startAsyncFuzzyFind(const SymbolIndex &Index,
                                            const FuzzyFindRequest &Req) {
  return runAsync<SymbolSlab>([&Index, Req]() {
    trace::Span Tracer("Async fuzzyFind");
    SymbolSlab::Builder Syms;
    Index.fuzzyFind(Req, [&Syms](const Symbol &Sym) { Syms.insert(Sym); });
    return std::move(Syms).build();
  });
}

// Creates a `FuzzyFindRequest` based on the cached index request from the
// last completion, if any, and the speculated completion filter text in the
// source code.
FuzzyFindRequest speculativeFuzzyFindRequestForCompletion(
    FuzzyFindRequest CachedReq, const CompletionPrefix &HeuristicPrefix) {
  CachedReq.Query = std::string(HeuristicPrefix.Name);
  return CachedReq;
}

// Runs Sema-based (AST) and Index-based completion, returns merged results.
//
// There are a few tricky considerations:
//   - the AST provides information needed for the index query (e.g. which
//     namespaces to search in). So Sema must start first.
//   - we only want to return the top results (Opts.Limit).
//     Building CompletionItems for everything else is wasteful, so we want to
//     preserve the "native" format until we're done with scoring.
//   - the data underlying Sema completion items is owned by the AST and various
//     other arenas, which must stay alive for us to build CompletionItems.
//   - we may get duplicate results from Sema and the Index, we need to merge.
//
// So we start Sema completion first, and do all our work in its callback.
// We use the Sema context information to query the index.
// Then we merge the two result sets, producing items that are Sema/Index/Both.
// These items are scored, and the top N are synthesized into the LSP response.
// Finally, we can clean up the data structures created by Sema completion.
//
// Main collaborators are:
//   - semaCodeComplete sets up the compiler machinery to run code completion.
//   - CompletionRecorder captures Sema completion results, including context.
//   - SymbolIndex (Opts.Index) provides index completion results as Symbols
//   - CompletionCandidates are the result of merging Sema and Index results.
//     Each candidate points to an underlying CodeCompletionResult (Sema), a
//     Symbol (Index), or both. It computes the result quality score.
//     CompletionCandidate also does conversion to CompletionItem (at the end).
//   - FuzzyMatcher scores how the candidate matches the partial identifier.
//     This score is combined with the result quality score for the final score.
//   - TopN determines the results with the best score.
class CodeCompleteFlow {
  PathRef FileName;
  IncludeStructure Includes;           // Complete once the compiler runs.
  SpeculativeFuzzyFind *SpecFuzzyFind; // Can be nullptr.
  const CodeCompleteOptions &Opts;

  // Sema takes ownership of Recorder. Recorder is valid until Sema cleanup.
  CompletionRecorder *Recorder = nullptr;
  CodeCompletionContext::Kind CCContextKind = CodeCompletionContext::CCC_Other;
  bool IsUsingDeclaration = false;
  // The snippets will not be generated if the token following completion
  // location is an opening parenthesis (tok::l_paren) because this would add
  // extra parenthesis.
  tok::TokenKind NextTokenKind = tok::eof;
  // Counters for logging.
  int NSema = 0, NIndex = 0, NSemaAndIndex = 0, NIdent = 0;
  bool Incomplete = false; // Would more be available with a higher limit?
  CompletionPrefix HeuristicPrefix;
  llvm::Optional<FuzzyMatcher> Filter; // Initialized once Sema runs.
  Range ReplacedRange;
  std::vector<std::string> QueryScopes; // Initialized once Sema runs.
  // Initialized once QueryScopes is initialized, if there are scopes.
  llvm::Optional<ScopeDistance> ScopeProximity;
  llvm::Optional<OpaqueType> PreferredType; // Initialized once Sema runs.
  // Whether to query symbols from any scope. Initialized once Sema runs.
  bool AllScopes = false;
  llvm::StringSet<> ContextWords;
  // Include-insertion and proximity scoring rely on the include structure.
  // This is available after Sema has run.
  llvm::Optional<IncludeInserter> Inserter;  // Available during runWithSema.
  llvm::Optional<URIDistance> FileProximity; // Initialized once Sema runs.
  /// Speculative request based on the cached request and the filter text before
  /// the cursor.
  /// Initialized right before sema run. This is only set if `SpecFuzzyFind` is
  /// set and contains a cached request.
  llvm::Optional<FuzzyFindRequest> SpecReq;

public:
  // A CodeCompleteFlow object is only useful for calling run() exactly once.
  CodeCompleteFlow(PathRef FileName, const IncludeStructure &Includes,
                   SpeculativeFuzzyFind *SpecFuzzyFind,
                   const CodeCompleteOptions &Opts)
      : FileName(FileName), Includes(Includes), SpecFuzzyFind(SpecFuzzyFind),
        Opts(Opts) {}

  CodeCompleteResult run(const SemaCompleteInput &SemaCCInput) && {
    trace::Span Tracer("CodeCompleteFlow");
    HeuristicPrefix = guessCompletionPrefix(SemaCCInput.ParseInput.Contents,
                                            SemaCCInput.Offset);
    populateContextWords(SemaCCInput.ParseInput.Contents);
    if (Opts.Index && SpecFuzzyFind && SpecFuzzyFind->CachedReq) {
      assert(!SpecFuzzyFind->Result.valid());
      SpecReq = speculativeFuzzyFindRequestForCompletion(
          *SpecFuzzyFind->CachedReq, HeuristicPrefix);
      SpecFuzzyFind->Result = startAsyncFuzzyFind(*Opts.Index, *SpecReq);
    }

    // We run Sema code completion first. It builds an AST and calculates:
    //   - completion results based on the AST.
    //   - partial identifier and context. We need these for the index query.
    CodeCompleteResult Output;
    auto RecorderOwner = std::make_unique<CompletionRecorder>(Opts, [&]() {
      assert(Recorder && "Recorder is not set");
      CCContextKind = Recorder->CCContext.getKind();
      IsUsingDeclaration = Recorder->CCContext.isUsingDeclaration();
      auto Style = getFormatStyleForFile(SemaCCInput.FileName,
                                         SemaCCInput.ParseInput.Contents,
                                         *SemaCCInput.ParseInput.TFS);
      const auto NextToken = Lexer::findNextToken(
          Recorder->CCSema->getPreprocessor().getCodeCompletionLoc(),
          Recorder->CCSema->getSourceManager(), Recorder->CCSema->LangOpts);
      if (NextToken)
        NextTokenKind = NextToken->getKind();
      // If preprocessor was run, inclusions from preprocessor callback should
      // already be added to Includes.
      Inserter.emplace(
          SemaCCInput.FileName, SemaCCInput.ParseInput.Contents, Style,
          SemaCCInput.ParseInput.CompileCommand.Directory,
          &Recorder->CCSema->getPreprocessor().getHeaderSearchInfo());
      for (const auto &Inc : Includes.MainFileIncludes)
        Inserter->addExisting(Inc);

      // Most of the cost of file proximity is in initializing the FileDistance
      // structures based on the observed includes, once per query. Conceptually
      // that happens here (though the per-URI-scheme initialization is lazy).
      // The per-result proximity scoring is (amortized) very cheap.
      FileDistanceOptions ProxOpts{}; // Use defaults.
      const auto &SM = Recorder->CCSema->getSourceManager();
      llvm::StringMap<SourceParams> ProxSources;
      auto MainFileID =
          Includes.getID(SM.getFileEntryForID(SM.getMainFileID()));
      assert(MainFileID);
      for (auto &HeaderIDAndDepth : Includes.includeDepth(*MainFileID)) {
        auto &Source =
            ProxSources[Includes.getRealPath(HeaderIDAndDepth.getFirst())];
        Source.Cost = HeaderIDAndDepth.getSecond() * ProxOpts.IncludeCost;
        // Symbols near our transitive includes are good, but only consider
        // things in the same directory or below it. Otherwise there can be
        // many false positives.
        if (HeaderIDAndDepth.getSecond() > 0)
          Source.MaxUpTraversals = 1;
      }
      FileProximity.emplace(ProxSources, ProxOpts);

      Output = runWithSema();
      Inserter.reset(); // Make sure this doesn't out-live Clang.
      SPAN_ATTACH(Tracer, "sema_completion_kind",
                  getCompletionKindString(CCContextKind));
      log("Code complete: sema context {0}, query scopes [{1}] (AnyScope={2}), "
          "expected type {3}{4}",
          getCompletionKindString(CCContextKind),
          llvm::join(QueryScopes.begin(), QueryScopes.end(), ","), AllScopes,
          PreferredType ? Recorder->CCContext.getPreferredType().getAsString()
                        : "<none>",
          IsUsingDeclaration ? ", inside using declaration" : "");
    });

    Recorder = RecorderOwner.get();

    semaCodeComplete(std::move(RecorderOwner), Opts.getClangCompleteOpts(),
                     SemaCCInput, &Includes);
    logResults(Output, Tracer);
    return Output;
  }

  void logResults(const CodeCompleteResult &Output, const trace::Span &Tracer) {
    SPAN_ATTACH(Tracer, "sema_results", NSema);
    SPAN_ATTACH(Tracer, "index_results", NIndex);
    SPAN_ATTACH(Tracer, "merged_results", NSemaAndIndex);
    SPAN_ATTACH(Tracer, "identifier_results", NIdent);
    SPAN_ATTACH(Tracer, "returned_results", int64_t(Output.Completions.size()));
    SPAN_ATTACH(Tracer, "incomplete", Output.HasMore);
    log("Code complete: {0} results from Sema, {1} from Index, "
        "{2} matched, {3} from identifiers, {4} returned{5}.",
        NSema, NIndex, NSemaAndIndex, NIdent, Output.Completions.size(),
        Output.HasMore ? " (incomplete)" : "");
    assert(!Opts.Limit || Output.Completions.size() <= Opts.Limit);
    // We don't assert that isIncomplete means we hit a limit.
    // Indexes may choose to impose their own limits even if we don't have one.
  }

  CodeCompleteResult runWithoutSema(llvm::StringRef Content, size_t Offset,
                                    const ThreadsafeFS &TFS) && {
    trace::Span Tracer("CodeCompleteWithoutSema");
    // Fill in fields normally set by runWithSema()
    HeuristicPrefix = guessCompletionPrefix(Content, Offset);
    populateContextWords(Content);
    CCContextKind = CodeCompletionContext::CCC_Recovery;
    IsUsingDeclaration = false;
    Filter = FuzzyMatcher(HeuristicPrefix.Name);
    auto Pos = offsetToPosition(Content, Offset);
    ReplacedRange.start = ReplacedRange.end = Pos;
    ReplacedRange.start.character -= HeuristicPrefix.Name.size();

    llvm::StringMap<SourceParams> ProxSources;
    ProxSources[FileName].Cost = 0;
    FileProximity.emplace(ProxSources);

    auto Style = getFormatStyleForFile(FileName, Content, TFS);
    // This will only insert verbatim headers.
    Inserter.emplace(FileName, Content, Style,
                     /*BuildDir=*/"", /*HeaderSearchInfo=*/nullptr);

    auto Identifiers = collectIdentifiers(Content, Style);
    std::vector<RawIdentifier> IdentifierResults;
    for (const auto &IDAndCount : Identifiers) {
      RawIdentifier ID;
      ID.Name = IDAndCount.first();
      ID.References = IDAndCount.second;
      // Avoid treating typed filter as an identifier.
      if (ID.Name == HeuristicPrefix.Name)
        --ID.References;
      if (ID.References > 0)
        IdentifierResults.push_back(std::move(ID));
    }

    // Simplified version of getQueryScopes():
    //  - accessible scopes are determined heuristically.
    //  - all-scopes query if no qualifier was typed (and it's allowed).
    SpecifiedScope Scopes;
    Scopes.AccessibleScopes = visibleNamespaces(
        Content.take_front(Offset), format::getFormattingLangOpts(Style));
    for (std::string &S : Scopes.AccessibleScopes)
      if (!S.empty())
        S.append("::"); // visibleNamespaces doesn't include trailing ::.
    if (HeuristicPrefix.Qualifier.empty())
      AllScopes = Opts.AllScopes;
    else if (HeuristicPrefix.Qualifier.startswith("::")) {
      Scopes.AccessibleScopes = {""};
      Scopes.UnresolvedQualifier =
          std::string(HeuristicPrefix.Qualifier.drop_front(2));
    } else
      Scopes.UnresolvedQualifier = std::string(HeuristicPrefix.Qualifier);
    // First scope is the (modified) enclosing scope.
    QueryScopes = Scopes.scopesForIndexQuery();
    ScopeProximity.emplace(QueryScopes);

    SymbolSlab IndexResults = Opts.Index ? queryIndex() : SymbolSlab();

    CodeCompleteResult Output = toCodeCompleteResult(mergeResults(
        /*SemaResults=*/{}, IndexResults, IdentifierResults));
    Output.RanParser = false;
    logResults(Output, Tracer);
    return Output;
  }

private:
  void populateContextWords(llvm::StringRef Content) {
    // Take last 3 lines before the completion point.
    unsigned RangeEnd = HeuristicPrefix.Qualifier.begin() - Content.data(),
             RangeBegin = RangeEnd;
    for (size_t I = 0; I < 3 && RangeBegin > 0; ++I) {
      auto PrevNL = Content.rfind('\n', RangeBegin);
      if (PrevNL == StringRef::npos) {
        RangeBegin = 0;
        break;
      }
      RangeBegin = PrevNL;
    }

    ContextWords = collectWords(Content.slice(RangeBegin, RangeEnd));
    dlog("Completion context words: {0}",
         llvm::join(ContextWords.keys(), ", "));
  }

  // This is called by run() once Sema code completion is done, but before the
  // Sema data structures are torn down. It does all the real work.
  CodeCompleteResult runWithSema() {
    const auto &CodeCompletionRange = CharSourceRange::getCharRange(
        Recorder->CCSema->getPreprocessor().getCodeCompletionTokenRange());
    // When we are getting completions with an empty identifier, for example
    //    std::vector<int> asdf;
    //    asdf.^;
    // Then the range will be invalid and we will be doing insertion, use
    // current cursor position in such cases as range.
    if (CodeCompletionRange.isValid()) {
      ReplacedRange = halfOpenToRange(Recorder->CCSema->getSourceManager(),
                                      CodeCompletionRange);
    } else {
      const auto &Pos = sourceLocToPosition(
          Recorder->CCSema->getSourceManager(),
          Recorder->CCSema->getPreprocessor().getCodeCompletionLoc());
      ReplacedRange.start = ReplacedRange.end = Pos;
    }
    Filter = FuzzyMatcher(
        Recorder->CCSema->getPreprocessor().getCodeCompletionFilter());
    std::tie(QueryScopes, AllScopes) = getQueryScopes(
        Recorder->CCContext, *Recorder->CCSema, HeuristicPrefix, Opts);
    if (!QueryScopes.empty())
      ScopeProximity.emplace(QueryScopes);
    PreferredType =
        OpaqueType::fromType(Recorder->CCSema->getASTContext(),
                             Recorder->CCContext.getPreferredType());
    // Sema provides the needed context to query the index.
    // FIXME: in addition to querying for extra/overlapping symbols, we should
    //        explicitly request symbols corresponding to Sema results.
    //        We can use their signals even if the index can't suggest them.
    // We must copy index results to preserve them, but there are at most Limit.
    auto IndexResults = (Opts.Index && allowIndex(Recorder->CCContext))
                            ? queryIndex()
                            : SymbolSlab();
    trace::Span Tracer("Populate CodeCompleteResult");
    // Merge Sema and Index results, score them, and pick the winners.
    auto Top =
        mergeResults(Recorder->Results, IndexResults, /*Identifiers*/ {});
    return toCodeCompleteResult(Top);
  }

  CodeCompleteResult
  toCodeCompleteResult(const std::vector<ScoredBundle> &Scored) {
    CodeCompleteResult Output;

    // Convert the results to final form, assembling the expensive strings.
    for (auto &C : Scored) {
      Output.Completions.push_back(toCodeCompletion(C.first));
      Output.Completions.back().Score = C.second;
      Output.Completions.back().CompletionTokenRange = ReplacedRange;
    }
    Output.HasMore = Incomplete;
    Output.Context = CCContextKind;
    Output.CompletionRange = ReplacedRange;
    return Output;
  }

  bool includeSymbolFromIndex(const Symbol &Sym) {
    if (CCContextKind == CodeCompletionContext::CCC_ObjCProtocolName) {
      return Sym.SymInfo.Lang == index::SymbolLanguage::ObjC &&
          Sym.SymInfo.Kind == index::SymbolKind::Protocol;
    }
    return true;
  }

  SymbolSlab queryIndex() {
    trace::Span Tracer("Query index");
    SPAN_ATTACH(Tracer, "limit", int64_t(Opts.Limit));

    // Build the query.
    FuzzyFindRequest Req;
    if (Opts.Limit)
      Req.Limit = Opts.Limit;
    Req.Query = std::string(Filter->pattern());
    Req.RestrictForCodeCompletion = true;
    Req.Scopes = QueryScopes;
    Req.AnyScope = AllScopes;
    // FIXME: we should send multiple weighted paths here.
    Req.ProximityPaths.push_back(std::string(FileName));
    if (PreferredType)
      Req.PreferredTypes.push_back(std::string(PreferredType->raw()));
    vlog("Code complete: fuzzyFind({0:2})", toJSON(Req));

    if (SpecFuzzyFind)
      SpecFuzzyFind->NewReq = Req;
    if (SpecFuzzyFind && SpecFuzzyFind->Result.valid() && (*SpecReq == Req)) {
      vlog("Code complete: speculative fuzzy request matches the actual index "
           "request. Waiting for the speculative index results.");
      SPAN_ATTACH(Tracer, "Speculative results", true);

      trace::Span WaitSpec("Wait speculative results");
      return SpecFuzzyFind->Result.get();
    }

    SPAN_ATTACH(Tracer, "Speculative results", false);

    // Run the query against the index.
    SymbolSlab::Builder ResultsBuilder;
    if (Opts.Index->fuzzyFind(Req, [&](const Symbol &Sym) {
          if (includeSymbolFromIndex(Sym))
            ResultsBuilder.insert(Sym);
        }))
      Incomplete = true;
    return std::move(ResultsBuilder).build();
  }

  // Merges Sema and Index results where possible, to form CompletionCandidates.
  // \p Identifiers is raw identifiers that can also be completion candidates.
  // Identifiers are not merged with results from index or sema.
  // Groups overloads if desired, to form CompletionCandidate::Bundles. The
  // bundles are scored and top results are returned, best to worst.
  std::vector<ScoredBundle>
  mergeResults(const std::vector<CodeCompletionResult> &SemaResults,
               const SymbolSlab &IndexResults,
               const std::vector<RawIdentifier> &IdentifierResults) {
    trace::Span Tracer("Merge and score results");
    std::vector<CompletionCandidate::Bundle> Bundles;
    llvm::DenseMap<size_t, size_t> BundleLookup;
    auto AddToBundles = [&](const CodeCompletionResult *SemaResult,
                            const Symbol *IndexResult,
                            const RawIdentifier *IdentifierResult) {
      CompletionCandidate C;
      C.SemaResult = SemaResult;
      C.IndexResult = IndexResult;
      C.IdentifierResult = IdentifierResult;
      if (C.IndexResult) {
        C.Name = IndexResult->Name;
        C.RankedIncludeHeaders = getRankedIncludes(*C.IndexResult);
      } else if (C.SemaResult) {
        C.Name = Recorder->getName(*SemaResult);
      } else {
        assert(IdentifierResult);
        C.Name = IdentifierResult->Name;
      }
      if (auto OverloadSet = C.overloadSet(
              Opts, FileName, Inserter ? Inserter.getPointer() : nullptr)) {
        auto Ret = BundleLookup.try_emplace(OverloadSet, Bundles.size());
        if (Ret.second)
          Bundles.emplace_back();
        Bundles[Ret.first->second].push_back(std::move(C));
      } else {
        Bundles.emplace_back();
        Bundles.back().push_back(std::move(C));
      }
    };
    llvm::DenseSet<const Symbol *> UsedIndexResults;
    auto CorrespondingIndexResult =
        [&](const CodeCompletionResult &SemaResult) -> const Symbol * {
      if (auto SymID =
              getSymbolID(SemaResult, Recorder->CCSema->getSourceManager())) {
        auto I = IndexResults.find(SymID);
        if (I != IndexResults.end()) {
          UsedIndexResults.insert(&*I);
          return &*I;
        }
      }
      return nullptr;
    };
    // Emit all Sema results, merging them with Index results if possible.
    for (auto &SemaResult : SemaResults)
      AddToBundles(&SemaResult, CorrespondingIndexResult(SemaResult), nullptr);
    // Now emit any Index-only results.
    for (const auto &IndexResult : IndexResults) {
      if (UsedIndexResults.count(&IndexResult))
        continue;
      AddToBundles(/*SemaResult=*/nullptr, &IndexResult, nullptr);
    }
    // Emit identifier results.
    for (const auto &Ident : IdentifierResults)
      AddToBundles(/*SemaResult=*/nullptr, /*IndexResult=*/nullptr, &Ident);
    // We only keep the best N results at any time, in "native" format.
    TopN<ScoredBundle, ScoredBundleGreater> Top(
        Opts.Limit == 0 ? std::numeric_limits<size_t>::max() : Opts.Limit);
    for (auto &Bundle : Bundles)
      addCandidate(Top, std::move(Bundle));
    return std::move(Top).items();
  }

  llvm::Optional<float> fuzzyScore(const CompletionCandidate &C) {
    // Macros can be very spammy, so we only support prefix completion.
    if (((C.SemaResult &&
          C.SemaResult->Kind == CodeCompletionResult::RK_Macro) ||
         (C.IndexResult &&
          C.IndexResult->SymInfo.Kind == index::SymbolKind::Macro)) &&
        !C.Name.startswith_insensitive(Filter->pattern()))
      return None;
    return Filter->match(C.Name);
  }

  CodeCompletion::Scores
  evaluateCompletion(const SymbolQualitySignals &Quality,
                     const SymbolRelevanceSignals &Relevance) {
    using RM = CodeCompleteOptions::CodeCompletionRankingModel;
    CodeCompletion::Scores Scores;
    switch (Opts.RankingModel) {
    case RM::Heuristics:
      Scores.Quality = Quality.evaluateHeuristics();
      Scores.Relevance = Relevance.evaluateHeuristics();
      Scores.Total =
          evaluateSymbolAndRelevance(Scores.Quality, Scores.Relevance);
      // NameMatch is in fact a multiplier on total score, so rescoring is
      // sound.
      Scores.ExcludingName =
          Relevance.NameMatch > std::numeric_limits<float>::epsilon()
              ? Scores.Total / Relevance.NameMatch
              : Scores.Quality;
      return Scores;

    case RM::DecisionForest:
      DecisionForestScores DFScores = Opts.DecisionForestScorer(
          Quality, Relevance, Opts.DecisionForestBase);
      Scores.ExcludingName = DFScores.ExcludingName;
      Scores.Total = DFScores.Total;
      return Scores;
    }
    llvm_unreachable("Unhandled CodeCompletion ranking model.");
  }

  // Scores a candidate and adds it to the TopN structure.
  void addCandidate(TopN<ScoredBundle, ScoredBundleGreater> &Candidates,
                    CompletionCandidate::Bundle Bundle) {
    SymbolQualitySignals Quality;
    SymbolRelevanceSignals Relevance;
    Relevance.Context = CCContextKind;
    Relevance.Name = Bundle.front().Name;
    Relevance.FilterLength = HeuristicPrefix.Name.size();
    Relevance.Query = SymbolRelevanceSignals::CodeComplete;
    Relevance.FileProximityMatch = FileProximity.getPointer();
    if (ScopeProximity)
      Relevance.ScopeProximityMatch = ScopeProximity.getPointer();
    if (PreferredType)
      Relevance.HadContextType = true;
    Relevance.ContextWords = &ContextWords;
    Relevance.MainFileSignals = Opts.MainFileSignals;

    auto &First = Bundle.front();
    if (auto FuzzyScore = fuzzyScore(First))
      Relevance.NameMatch = *FuzzyScore;
    else
      return;
    SymbolOrigin Origin = SymbolOrigin::Unknown;
    bool FromIndex = false;
    for (const auto &Candidate : Bundle) {
      if (Candidate.IndexResult) {
        Quality.merge(*Candidate.IndexResult);
        Relevance.merge(*Candidate.IndexResult);
        Origin |= Candidate.IndexResult->Origin;
        FromIndex = true;
        if (!Candidate.IndexResult->Type.empty())
          Relevance.HadSymbolType |= true;
        if (PreferredType &&
            PreferredType->raw() == Candidate.IndexResult->Type) {
          Relevance.TypeMatchesPreferred = true;
        }
      }
      if (Candidate.SemaResult) {
        Quality.merge(*Candidate.SemaResult);
        Relevance.merge(*Candidate.SemaResult);
        if (PreferredType) {
          if (auto CompletionType = OpaqueType::fromCompletionResult(
                  Recorder->CCSema->getASTContext(), *Candidate.SemaResult)) {
            Relevance.HadSymbolType |= true;
            if (PreferredType == CompletionType)
              Relevance.TypeMatchesPreferred = true;
          }
        }
        Origin |= SymbolOrigin::AST;
      }
      if (Candidate.IdentifierResult) {
        Quality.References = Candidate.IdentifierResult->References;
        Relevance.Scope = SymbolRelevanceSignals::FileScope;
        Origin |= SymbolOrigin::Identifier;
      }
    }

    CodeCompletion::Scores Scores = evaluateCompletion(Quality, Relevance);
    if (Opts.RecordCCResult)
      Opts.RecordCCResult(toCodeCompletion(Bundle), Quality, Relevance,
                          Scores.Total);

    dlog("CodeComplete: {0} ({1}) = {2}\n{3}{4}\n", First.Name,
         llvm::to_string(Origin), Scores.Total, llvm::to_string(Quality),
         llvm::to_string(Relevance));

    NSema += bool(Origin & SymbolOrigin::AST);
    NIndex += FromIndex;
    NSemaAndIndex += bool(Origin & SymbolOrigin::AST) && FromIndex;
    NIdent += bool(Origin & SymbolOrigin::Identifier);
    if (Candidates.push({std::move(Bundle), Scores}))
      Incomplete = true;
  }

  CodeCompletion toCodeCompletion(const CompletionCandidate::Bundle &Bundle) {
    llvm::Optional<CodeCompletionBuilder> Builder;
    for (const auto &Item : Bundle) {
      CodeCompletionString *SemaCCS =
          Item.SemaResult ? Recorder->codeCompletionString(*Item.SemaResult)
                          : nullptr;
      if (!Builder)
        Builder.emplace(Recorder ? &Recorder->CCSema->getASTContext() : nullptr,
                        Item, SemaCCS, QueryScopes, *Inserter, FileName,
                        CCContextKind, Opts, IsUsingDeclaration, NextTokenKind);
      else
        Builder->add(Item, SemaCCS);
    }
    return Builder->build();
  }
};

} // namespace

clang::CodeCompleteOptions CodeCompleteOptions::getClangCompleteOpts() const {
  clang::CodeCompleteOptions Result;
  Result.IncludeCodePatterns = EnableSnippets;
  Result.IncludeMacros = true;
  Result.IncludeGlobals = true;
  // We choose to include full comments and not do doxygen parsing in
  // completion.
  // FIXME: ideally, we should support doxygen in some form, e.g. do markdown
  // formatting of the comments.
  Result.IncludeBriefComments = false;

  // When an is used, Sema is responsible for completing the main file,
  // the index can provide results from the preamble.
  // Tell Sema not to deserialize the preamble to look for results.
  Result.LoadExternal = !Index;
  Result.IncludeFixIts = IncludeFixIts;

  return Result;
}

CompletionPrefix guessCompletionPrefix(llvm::StringRef Content,
                                       unsigned Offset) {
  assert(Offset <= Content.size());
  StringRef Rest = Content.take_front(Offset);
  CompletionPrefix Result;

  // Consume the unqualified name. We only handle ASCII characters.
  // isAsciiIdentifierContinue will let us match "0invalid", but we don't mind.
  while (!Rest.empty() && isAsciiIdentifierContinue(Rest.back()))
    Rest = Rest.drop_back();
  Result.Name = Content.slice(Rest.size(), Offset);

  // Consume qualifiers.
  while (Rest.consume_back("::") && !Rest.endswith(":")) // reject ::::
    while (!Rest.empty() && isAsciiIdentifierContinue(Rest.back()))
      Rest = Rest.drop_back();
  Result.Qualifier =
      Content.slice(Rest.size(), Result.Name.begin() - Content.begin());

  return Result;
}

// Code complete the argument name on "/*" inside function call.
// Offset should be pointing to the start of the comment, i.e.:
// foo(^/*, rather than foo(/*^) where the cursor probably is.
CodeCompleteResult codeCompleteComment(PathRef FileName, unsigned Offset,
                                       llvm::StringRef Prefix,
                                       const PreambleData *Preamble,
                                       const ParseInputs &ParseInput) {
  if (Preamble == nullptr) // Can't run without Sema.
    return CodeCompleteResult();

  clang::CodeCompleteOptions Options;
  Options.IncludeGlobals = false;
  Options.IncludeMacros = false;
  Options.IncludeCodePatterns = false;
  Options.IncludeBriefComments = false;
  std::set<std::string> ParamNames;
  // We want to see signatures coming from newly introduced includes, hence a
  // full patch.
  semaCodeComplete(
      std::make_unique<ParamNameCollector>(Options, ParamNames), Options,
      {FileName, Offset, *Preamble,
       PreamblePatch::createFullPatch(FileName, ParseInput, *Preamble),
       ParseInput});
  if (ParamNames.empty())
    return CodeCompleteResult();

  CodeCompleteResult Result;
  Result.Context = CodeCompletionContext::CCC_NaturalLanguage;
  for (llvm::StringRef Name : ParamNames) {
    if (!Name.startswith(Prefix))
      continue;
    CodeCompletion Item;
    Item.Name = Name.str() + "=";
    Item.FilterText = Item.Name;
    Item.Kind = CompletionItemKind::Text;
    Result.Completions.push_back(Item);
  }

  return Result;
}

// If Offset is inside what looks like argument comment (e.g.
// "/*^" or "/* foo^"), returns new offset pointing to the start of the /*
// (place where semaCodeComplete should run).
llvm::Optional<unsigned>
maybeFunctionArgumentCommentStart(llvm::StringRef Content) {
  while (!Content.empty() && isAsciiIdentifierContinue(Content.back()))
    Content = Content.drop_back();
  Content = Content.rtrim();
  if (Content.endswith("/*"))
    return Content.size() - 2;
  return None;
}

CodeCompleteResult codeComplete(PathRef FileName, Position Pos,
                                const PreambleData *Preamble,
                                const ParseInputs &ParseInput,
                                CodeCompleteOptions Opts,
                                SpeculativeFuzzyFind *SpecFuzzyFind) {
  auto Offset = positionToOffset(ParseInput.Contents, Pos);
  if (!Offset) {
    elog("Code completion position was invalid {0}", Offset.takeError());
    return CodeCompleteResult();
  }

  auto Content = llvm::StringRef(ParseInput.Contents).take_front(*Offset);
  if (auto OffsetBeforeComment = maybeFunctionArgumentCommentStart(Content)) {
    // We are doing code completion of a comment, where we currently only
    // support completing param names in function calls. To do this, we
    // require information from Sema, but Sema's comment completion stops at
    // parsing, so we must move back the position before running it, extract
    // information we need and construct completion items ourselves.
    auto CommentPrefix = Content.substr(*OffsetBeforeComment + 2).trim();
    return codeCompleteComment(FileName, *OffsetBeforeComment, CommentPrefix,
                               Preamble, ParseInput);
  }

  auto Flow = CodeCompleteFlow(
      FileName, Preamble ? Preamble->Includes : IncludeStructure(),
      SpecFuzzyFind, Opts);
  return (!Preamble || Opts.RunParser == CodeCompleteOptions::NeverParse)
             ? std::move(Flow).runWithoutSema(ParseInput.Contents, *Offset,
                                              *ParseInput.TFS)
             : std::move(Flow).run({FileName, *Offset, *Preamble,
                                    /*PreamblePatch=*/
                                    PreamblePatch::createMacroPatch(
                                        FileName, ParseInput, *Preamble),
                                    ParseInput});
}

SignatureHelp signatureHelp(PathRef FileName, Position Pos,
                            const PreambleData &Preamble,
                            const ParseInputs &ParseInput,
                            MarkupKind DocumentationFormat) {
  auto Offset = positionToOffset(ParseInput.Contents, Pos);
  if (!Offset) {
    elog("Signature help position was invalid {0}", Offset.takeError());
    return SignatureHelp();
  }
  SignatureHelp Result;
  clang::CodeCompleteOptions Options;
  Options.IncludeGlobals = false;
  Options.IncludeMacros = false;
  Options.IncludeCodePatterns = false;
  Options.IncludeBriefComments = false;
  semaCodeComplete(
      std::make_unique<SignatureHelpCollector>(Options, DocumentationFormat,
                                               ParseInput.Index, Result),
      Options,
      {FileName, *Offset, Preamble,
       PreamblePatch::createFullPatch(FileName, ParseInput, Preamble),
       ParseInput});
  return Result;
}

bool isIndexedForCodeCompletion(const NamedDecl &ND, ASTContext &ASTCtx) {
  auto InTopLevelScope = [](const NamedDecl &ND) {
    switch (ND.getDeclContext()->getDeclKind()) {
    case Decl::TranslationUnit:
    case Decl::Namespace:
    case Decl::LinkageSpec:
      return true;
    default:
      break;
    };
    return false;
  };
  // We only complete symbol's name, which is the same as the name of the
  // *primary* template in case of template specializations.
  if (isExplicitTemplateSpecialization(&ND))
    return false;

  // Category decls are not useful on their own outside the interface or
  // implementation blocks. Moreover, sema already provides completion for
  // these, even if it requires preamble deserialization. So by excluding them
  // from the index, we reduce the noise in all the other completion scopes.
  if (llvm::isa<ObjCCategoryDecl>(&ND) || llvm::isa<ObjCCategoryImplDecl>(&ND))
    return false;

  if (InTopLevelScope(ND))
    return true;

  if (const auto *EnumDecl = dyn_cast<clang::EnumDecl>(ND.getDeclContext()))
    return InTopLevelScope(*EnumDecl) && !EnumDecl->isScoped();

  return false;
}

CompletionItem CodeCompletion::render(const CodeCompleteOptions &Opts) const {
  CompletionItem LSP;
  const auto *InsertInclude = Includes.empty() ? nullptr : &Includes[0];
  LSP.label = ((InsertInclude && InsertInclude->Insertion)
                   ? Opts.IncludeIndicator.Insert
                   : Opts.IncludeIndicator.NoInsert) +
              (Opts.ShowOrigins ? "[" + llvm::to_string(Origin) + "]" : "") +
              RequiredQualifier + Name + Signature;

  LSP.kind = Kind;
  LSP.detail = BundleSize > 1
                   ? std::string(llvm::formatv("[{0} overloads]", BundleSize))
                   : ReturnType;
  LSP.deprecated = Deprecated;
  // Combine header information and documentation in LSP `documentation` field.
  // This is not quite right semantically, but tends to display well in editors.
  if (InsertInclude || Documentation) {
    markup::Document Doc;
    if (InsertInclude)
      Doc.addParagraph().appendText("From ").appendCode(InsertInclude->Header);
    if (Documentation)
      Doc.append(*Documentation);
    LSP.documentation = renderDoc(Doc, Opts.DocumentationFormat);
  }
  LSP.sortText = sortText(Score.Total, FilterText);
  LSP.filterText = FilterText;
  LSP.textEdit = {CompletionTokenRange, RequiredQualifier + Name};
  // Merge continuous additionalTextEdits into main edit. The main motivation
  // behind this is to help LSP clients, it seems most of them are confused when
  // they are provided with additionalTextEdits that are consecutive to main
  // edit.
  // Note that we store additional text edits from back to front in a line. That
  // is mainly to help LSP clients again, so that changes do not effect each
  // other.
  for (const auto &FixIt : FixIts) {
    if (FixIt.range.end == LSP.textEdit->range.start) {
      LSP.textEdit->newText = FixIt.newText + LSP.textEdit->newText;
      LSP.textEdit->range.start = FixIt.range.start;
    } else {
      LSP.additionalTextEdits.push_back(FixIt);
    }
  }
  if (Opts.EnableSnippets)
    LSP.textEdit->newText += SnippetSuffix;

  // FIXME(kadircet): Do not even fill insertText after making sure textEdit is
  // compatible with most of the editors.
  LSP.insertText = LSP.textEdit->newText;
  // Some clients support snippets but work better with plaintext.
  // So if the snippet is trivial, let the client know.
  // https://github.com/clangd/clangd/issues/922
  LSP.insertTextFormat = (Opts.EnableSnippets && !SnippetSuffix.empty())
                             ? InsertTextFormat::Snippet
                             : InsertTextFormat::PlainText;
  if (InsertInclude && InsertInclude->Insertion)
    LSP.additionalTextEdits.push_back(*InsertInclude->Insertion);

  LSP.score = Score.ExcludingName;

  return LSP;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const CodeCompletion &C) {
  // For now just lean on CompletionItem.
  return OS << C.render(CodeCompleteOptions());
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const CodeCompleteResult &R) {
  OS << "CodeCompleteResult: " << R.Completions.size() << (R.HasMore ? "+" : "")
     << " (" << getCompletionKindString(R.Context) << ")"
     << " items:\n";
  for (const auto &C : R.Completions)
    OS << C << "\n";
  return OS;
}

// Heuristically detect whether the `Line` is an unterminated include filename.
bool isIncludeFile(llvm::StringRef Line) {
  Line = Line.ltrim();
  if (!Line.consume_front("#"))
    return false;
  Line = Line.ltrim();
  if (!(Line.consume_front("include_next") || Line.consume_front("include") ||
        Line.consume_front("import")))
    return false;
  Line = Line.ltrim();
  if (Line.consume_front("<"))
    return Line.count('>') == 0;
  if (Line.consume_front("\""))
    return Line.count('"') == 0;
  return false;
}

bool allowImplicitCompletion(llvm::StringRef Content, unsigned Offset) {
  // Look at last line before completion point only.
  Content = Content.take_front(Offset);
  auto Pos = Content.rfind('\n');
  if (Pos != llvm::StringRef::npos)
    Content = Content.substr(Pos + 1);

  // Complete after scope operators.
  if (Content.endswith(".") || Content.endswith("->") ||
      Content.endswith("::") || Content.endswith("/*"))
    return true;
  // Complete after `#include <` and #include `<foo/`.
  if ((Content.endswith("<") || Content.endswith("\"") ||
       Content.endswith("/")) &&
      isIncludeFile(Content))
    return true;

  // Complete words. Give non-ascii characters the benefit of the doubt.
  return !Content.empty() && (isAsciiIdentifierContinue(Content.back()) ||
                              !llvm::isASCII(Content.back()));
}

} // namespace clangd
} // namespace clang
