//===--- CodeComplete.cpp ----------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "ClangdUnit.h"
#include "CodeCompletionStrings.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "ExpectedTypes.h"
#include "FileDistance.h"
#include "FuzzyMatch.h"
#include "Headers.h"
#include "Logger.h"
#include "Quality.h"
#include "SourceCode.h"
#include "TUScheduler.h"
#include "Trace.h"
#include "URI.h"
#include "index/Index.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ScopedPrinter.h"
#include <algorithm>
#include <iterator>

// We log detailed candidate here if you run with -debug-only=codecomplete.
#define DEBUG_TYPE "CodeComplete"

using namespace llvm;
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
  // FIXME(ioeric): use LSP struct instead of class when it is suppoted in the
  // protocol.
  case SK::Struct:
  case SK::Class:
  case SK::Protocol:
  case SK::Extension:
  case SK::Union:
    return CompletionItemKind::Class;
  // FIXME(ioeric): figure out whether reference is the right type for aliases.
  case SK::TypeAlias:
  case SK::Using:
    return CompletionItemKind::Reference;
  case SK::Function:
  // FIXME(ioeric): this should probably be an operator. This should be fixed
  // when `Operator` is support type in the protocol.
  case SK::ConversionFunction:
    return CompletionItemKind::Function;
  case SK::Variable:
  case SK::Parameter:
    return CompletionItemKind::Variable;
  case SK::Field:
    return CompletionItemKind::Field;
  // FIXME(ioeric): use LSP enum constant when it is supported in the protocol.
  case SK::EnumConstant:
    return CompletionItemKind::Value;
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

/// Get the optional chunk as a string. This function is possibly recursive.
///
/// The parameter info for each parameter is appended to the Parameters.
std::string getOptionalParameters(const CodeCompletionString &CCS,
                                  std::vector<ParameterInformation> &Parameters,
                                  SignatureQualitySignals &Signal) {
  std::string Result;
  for (const auto &Chunk : CCS) {
    switch (Chunk.Kind) {
    case CodeCompletionString::CK_Optional:
      assert(Chunk.Optional &&
             "Expected the optional code completion string to be non-null.");
      Result += getOptionalParameters(*Chunk.Optional, Parameters, Signal);
      break;
    case CodeCompletionString::CK_VerticalSpace:
      break;
    case CodeCompletionString::CK_Placeholder:
      // A string that acts as a placeholder for, e.g., a function call
      // argument.
      // Intentional fallthrough here.
    case CodeCompletionString::CK_CurrentParameter: {
      // A piece of text that describes the parameter that corresponds to
      // the code-completion location within a function call, message send,
      // macro invocation, etc.
      Result += Chunk.Text;
      ParameterInformation Info;
      Info.label = Chunk.Text;
      Parameters.push_back(std::move(Info));
      Signal.ContainsActiveParameter = true;
      Signal.NumberOfOptionalParameters++;
      break;
    }
    default:
      Result += Chunk.Text;
      break;
    }
  }
  return Result;
}

/// Creates a `HeaderFile` from \p Header which can be either a URI or a literal
/// include.
static Expected<HeaderFile> toHeaderFile(StringRef Header, StringRef HintPath) {
  if (isLiteralInclude(Header))
    return HeaderFile{Header.str(), /*Verbatim=*/true};
  auto U = URI::parse(Header);
  if (!U)
    return U.takeError();

  auto IncludePath = URI::includeSpelling(*U);
  if (!IncludePath)
    return IncludePath.takeError();
  if (!IncludePath->empty())
    return HeaderFile{std::move(*IncludePath), /*Verbatim=*/true};

  auto Resolved = URI::resolve(*U, HintPath);
  if (!Resolved)
    return Resolved.takeError();
  return HeaderFile{std::move(*Resolved), /*Verbatim=*/false};
}

/// A code completion result, in clang-native form.
/// It may be promoted to a CompletionItem if it's among the top-ranked results.
struct CompletionCandidate {
  StringRef Name; // Used for filtering and sorting.
  // We may have a result from Sema, from the index, or both.
  const CodeCompletionResult *SemaResult = nullptr;
  const Symbol *IndexResult = nullptr;
  SmallVector<StringRef, 1> RankedIncludeHeaders;

  // Returns a token identifying the overload set this is part of.
  // 0 indicates it's not part of any overload set.
  size_t overloadSet() const {
    SmallString<256> Scratch;
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
        return hash_combine(
            (IndexResult->Scope + IndexResult->Name).toStringRef(Scratch),
            headerToInsertIfAllowed().getValueOr(""));
      default:
        return 0;
      }
    }
    assert(SemaResult);
    // We need to make sure we're consistent with the IndexResult case!
    const NamedDecl *D = SemaResult->Declaration;
    if (!D || !D->isFunctionOrFunctionTemplate())
      return 0;
    {
      raw_svector_ostream OS(Scratch);
      D->printQualifiedName(OS);
    }
    return hash_combine(Scratch, headerToInsertIfAllowed().getValueOr(""));
  }

  // The best header to include if include insertion is allowed.
  Optional<StringRef> headerToInsertIfAllowed() const {
    if (RankedIncludeHeaders.empty())
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

  using Bundle = SmallVector<CompletionCandidate, 4>;
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
  CodeCompletionBuilder(ASTContext &ASTCtx, const CompletionCandidate &C,
                        CodeCompletionString *SemaCCS,
                        ArrayRef<std::string> QueryScopes,
                        const IncludeInserter &Includes, StringRef FileName,
                        CodeCompletionContext::Kind ContextKind,
                        const CodeCompleteOptions &Opts)
      : ASTCtx(ASTCtx), ExtractDocumentation(Opts.IncludeComments),
        EnableFunctionArgSnippets(Opts.EnableFunctionArgSnippets) {
    add(C, SemaCCS);
    if (C.SemaResult) {
      Completion.Origin |= SymbolOrigin::AST;
      Completion.Name = StringRef(SemaCCS->getTypedText());
      if (Completion.Scope.empty()) {
        if ((C.SemaResult->Kind == CodeCompletionResult::RK_Declaration) ||
            (C.SemaResult->Kind == CodeCompletionResult::RK_Pattern))
          if (const auto *D = C.SemaResult->getDeclaration())
            if (const auto *ND = dyn_cast<NamedDecl>(D))
              Completion.Scope =
                  splitQualifiedName(printQualifiedName(*ND)).first;
      }
      Completion.Kind = toCompletionItemKind(
          C.SemaResult->Kind, C.SemaResult->Declaration, ContextKind);
      // Sema could provide more info on whether the completion was a file or
      // folder.
      if (Completion.Kind == CompletionItemKind::File &&
          Completion.Name.back() == '/')
        Completion.Kind = CompletionItemKind::Folder;
      for (const auto &FixIt : C.SemaResult->FixIts) {
        Completion.FixIts.push_back(
            toTextEdit(FixIt, ASTCtx.getSourceManager(), ASTCtx.getLangOpts()));
      }
      llvm::sort(Completion.FixIts, [](const TextEdit &X, const TextEdit &Y) {
        return std::tie(X.range.start.line, X.range.start.character) <
               std::tie(Y.range.start.line, Y.range.start.character);
      });
      Completion.Deprecated |=
          (C.SemaResult->Availability == CXAvailability_Deprecated);
    }
    if (C.IndexResult) {
      Completion.Origin |= C.IndexResult->Origin;
      if (Completion.Scope.empty())
        Completion.Scope = C.IndexResult->Scope;
      if (Completion.Kind == CompletionItemKind::Missing)
        Completion.Kind = toCompletionItemKind(C.IndexResult->SymInfo.Kind);
      if (Completion.Name.empty())
        Completion.Name = C.IndexResult->Name;
      // If the completion was visible to Sema, no qualifier is needed. This
      // avoids unneeded qualifiers in cases like with `using ns::X`.
      if (Completion.RequiredQualifier.empty() && !C.SemaResult) {
        StringRef ShortestQualifier = C.IndexResult->Scope;
        for (StringRef Scope : QueryScopes) {
          StringRef Qualifier = C.IndexResult->Scope;
          if (Qualifier.consume_front(Scope) &&
              Qualifier.size() < ShortestQualifier.size())
            ShortestQualifier = Qualifier;
        }
        Completion.RequiredQualifier = ShortestQualifier;
      }
      Completion.Deprecated |= (C.IndexResult->Flags & Symbol::Deprecated);
    }

    // Turn absolute path into a literal string that can be #included.
    auto Inserted =
        [&](StringRef Header) -> Expected<std::pair<std::string, bool>> {
      auto ResolvedDeclaring =
          toHeaderFile(C.IndexResult->CanonicalDeclaration.FileURI, FileName);
      if (!ResolvedDeclaring)
        return ResolvedDeclaring.takeError();
      auto ResolvedInserted = toHeaderFile(Header, FileName);
      if (!ResolvedInserted)
        return ResolvedInserted.takeError();
      return std::make_pair(
          Includes.calculateIncludePath(*ResolvedDeclaring, *ResolvedInserted),
          Includes.shouldInsertInclude(*ResolvedDeclaring, *ResolvedInserted));
    };
    bool ShouldInsert = C.headerToInsertIfAllowed().hasValue();
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
            "(FileURI='{0}', IncludeHeader='{1}') into {2}",
            C.IndexResult->CanonicalDeclaration.FileURI, Inc, FileName);
    }
    // Prefer includes that do not need edits (i.e. already exist).
    std::stable_partition(Completion.Includes.begin(),
                          Completion.Includes.end(),
                          [](const CodeCompletion::IncludeCandidate &I) {
                            return !I.Insertion.hasValue();
                          });
  }

  void add(const CompletionCandidate &C, CodeCompletionString *SemaCCS) {
    assert(bool(C.SemaResult) == bool(SemaCCS));
    Bundled.emplace_back();
    BundledEntry &S = Bundled.back();
    if (C.SemaResult) {
      getSignature(*SemaCCS, &S.Signature, &S.SnippetSuffix,
                   &Completion.RequiredQualifier);
      S.ReturnType = getReturnType(*SemaCCS);
    } else if (C.IndexResult) {
      S.Signature = C.IndexResult->Signature;
      S.SnippetSuffix = C.IndexResult->CompletionSnippetSuffix;
      S.ReturnType = C.IndexResult->ReturnType;
    }
    if (ExtractDocumentation && Completion.Documentation.empty()) {
      if (C.IndexResult)
        Completion.Documentation = C.IndexResult->Documentation;
      else if (C.SemaResult)
        Completion.Documentation = getDocComment(ASTCtx, *C.SemaResult,
                                                 /*CommentsFromHeader=*/false);
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

  // If all BundledEntrys have the same value for a property, return it.
  template <std::string BundledEntry::*Member>
  const std::string *onlyValue() const {
    auto B = Bundled.begin(), E = Bundled.end();
    for (auto I = B + 1; I != E; ++I)
      if (I->*Member != B->*Member)
        return nullptr;
    return &(B->*Member);
  }

  template <bool BundledEntry::*Member> const bool *onlyValue() const {
    auto B = Bundled.begin(), E = Bundled.end();
    for (auto I = B + 1; I != E; ++I)
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
    auto *Snippet = onlyValue<&BundledEntry::SnippetSuffix>();
    if (!Snippet)
      // All bundles are function calls.
      // FIXME(ibiryukov): sometimes add template arguments to a snippet, e.g.
      // we need to complete 'forward<$1>($0)'.
      return "($0)";
    if (EnableFunctionArgSnippets)
      return *Snippet;

    // Replace argument snippets with a simplified pattern.
    if (Snippet->empty())
      return "";
    if (Completion.Kind == CompletionItemKind::Function ||
        Completion.Kind == CompletionItemKind::Method) {
      // Functions snippets can be of 2 types:
      // - containing only function arguments, e.g.
      //   foo(${1:int p1}, ${2:int p2});
      //   We transform this pattern to '($0)' or '()'.
      // - template arguments and function arguments, e.g.
      //   foo<${1:class}>(${2:int p1}).
      //   We transform this pattern to '<$1>()$0' or '<$0>()'.

      bool EmptyArgs = StringRef(*Snippet).endswith("()");
      if (Snippet->front() == '<')
        return EmptyArgs ? "<$1>()$0" : "<$1>($0)";
      if (Snippet->front() == '(')
        return EmptyArgs ? "()" : "($0)";
      return *Snippet; // Not an arg snippet?
    }
    if (Completion.Kind == CompletionItemKind::Reference ||
        Completion.Kind == CompletionItemKind::Class) {
      if (Snippet->front() != '<')
        return *Snippet; // Not an arg snippet?

      // Classes and template using aliases can only have template arguments,
      // e.g. Foo<${1:class}>.
      if (StringRef(*Snippet).endswith("<>"))
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

  ASTContext &ASTCtx;
  CodeCompletion Completion;
  SmallVector<BundledEntry, 1> Bundled;
  bool ExtractDocumentation;
  bool EnableFunctionArgSnippets;
};

// Determine the symbol ID for a Sema code completion result, if possible.
Optional<SymbolID> getSymbolID(const CodeCompletionResult &R,
                               const SourceManager &SM) {
  switch (R.Kind) {
  case CodeCompletionResult::RK_Declaration:
  case CodeCompletionResult::RK_Pattern: {
    return clang::clangd::getSymbolID(R.Declaration);
  }
  case CodeCompletionResult::RK_Macro:
    return clang::clangd::getSymbolID(*R.Macro, R.MacroDefInfo, SM);
  case CodeCompletionResult::RK_Keyword:
    return None;
  }
  llvm_unreachable("unknown CodeCompletionResult kind");
}

// Scopes of the paritial identifier we're trying to complete.
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
  Optional<std::string> UnresolvedQualifier;

  // Construct scopes being queried in indexes. The results are deduplicated.
  // This method format the scopes to match the index request representation.
  std::vector<std::string> scopesForIndexQuery() {
    std::set<std::string> Results;
    for (StringRef AS : AccessibleScopes)
      Results.insert(
          ((UnresolvedQualifier ? *UnresolvedQualifier : "") + AS).str());
    return {Results.begin(), Results.end()};
  }
};

// Get all scopes that will be queried in indexes and whether symbols from
// any scope is allowed. The first scope in the list is the preferred scope
// (e.g. enclosing namespace).
std::pair<std::vector<std::string>, bool>
getQueryScopes(CodeCompletionContext &CCContext, const Sema &CCSema,
               const CodeCompleteOptions &Opts) {
  auto GetAllAccessibleScopes = [](CodeCompletionContext &CCContext) {
    SpecifiedScope Info;
    for (auto *Context : CCContext.getVisitedContexts()) {
      if (isa<TranslationUnitDecl>(Context))
        Info.AccessibleScopes.push_back(""); // global namespace
      else if (isa<NamespaceDecl>(Context))
        Info.AccessibleScopes.push_back(printNamespaceScope(*Context));
    }
    return Info;
  };

  auto SS = CCContext.getCXXScopeSpecifier();

  // Unqualified completion (e.g. "vec^").
  if (!SS) {
    std::vector<std::string> Scopes;
    std::string EnclosingScope = printNamespaceScope(*CCSema.CurContext);
    Scopes.push_back(EnclosingScope);
    for (auto &S : GetAllAccessibleScopes(CCContext).scopesForIndexQuery()) {
      if (EnclosingScope != S)
        Scopes.push_back(std::move(S));
    }
    // Allow AllScopes completion only for there is no explicit scope qualifier.
    return {Scopes, Opts.AllScopes};
  }

  // Qualified completion ("std::vec^"), we have two cases depending on whether
  // the qualifier can be resolved by Sema.
  if ((*SS)->isValid()) { // Resolved qualifier.
    return {GetAllAccessibleScopes(CCContext).scopesForIndexQuery(), false};
  }

  // Unresolved qualifier.
  // FIXME: When Sema can resolve part of a scope chain (e.g.
  // "known::unknown::id"), we should expand the known part ("known::") rather
  // than treating the whole thing as unknown.
  SpecifiedScope Info;
  Info.AccessibleScopes.push_back(""); // global namespace

  Info.UnresolvedQualifier =
      Lexer::getSourceText(CharSourceRange::getCharRange((*SS)->getRange()),
                           CCSema.SourceMgr, clang::LangOptions())
          .ltrim("::");
  // Sema excludes the trailing "::".
  if (!Info.UnresolvedQualifier->empty())
    *Info.UnresolvedQualifier += "::";

  return {Info.scopesForIndexQuery(), false};
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

// Some member calls are blacklisted because they're so rarely useful.
static bool isBlacklistedMember(const NamedDecl &D) {
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
                     unique_function<void()> ResultsCallback)
      : CodeCompleteConsumer(Opts.getClangCompleteOpts(),
                             /*OutputIsBinary=*/false),
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
                                  unsigned NumResults) override final {
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
          && isBlacklistedMember(*Result.Declaration))
        continue;
      // Skip injected class name when no class scope is not explicitly set.
      // E.g. show injected A::A in `using A::A^` but not in "A^".
      if (Result.Declaration && !Context.getCXXScopeSpecifier().hasValue() &&
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
  StringRef getName(const CodeCompletionResult &Result) {
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
      return Result.Pattern->getTypedText();
    }
    auto *CCS = codeCompletionString(Result);
    return CCS->getTypedText();
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
  unique_function<void()> ResultsCallback;
};

struct ScoredSignature {
  // When set, requires documentation to be requested from the index with this
  // ID.
  Optional<SymbolID> IDForDoc;
  SignatureInformation Signature;
  SignatureQualitySignals Quality;
};

class SignatureHelpCollector final : public CodeCompleteConsumer {
public:
  SignatureHelpCollector(const clang::CodeCompleteOptions &CodeCompleteOpts,
                         const SymbolIndex *Index, SignatureHelp &SigHelp)
      : CodeCompleteConsumer(CodeCompleteOpts,
                             /*OutputIsBinary=*/false),
        SigHelp(SigHelp),
        Allocator(std::make_shared<clang::GlobalCodeCompletionAllocator>()),
        CCTUInfo(Allocator), Index(Index) {}

  void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                 OverloadCandidate *Candidates,
                                 unsigned NumCandidates,
                                 SourceLocation OpenParLoc) override {
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

      const auto *CCS = Candidate.CreateSignatureString(
          CurrentArg, S, *Allocator, CCTUInfo, true);
      assert(CCS && "Expected the CodeCompletionString to be non-null");
      ScoredSignatures.push_back(processOverloadCandidate(
          Candidate, *CCS,
          Candidate.getFunction()
              ? getDeclComment(S.getASTContext(), *Candidate.getFunction())
              : ""));
    }

    // Sema does not load the docs from the preamble, so we need to fetch extra
    // docs from the index instead.
    DenseMap<SymbolID, std::string> FetchedDocs;
    if (Index) {
      LookupRequest IndexRequest;
      for (const auto &S : ScoredSignatures) {
        if (!S.IDForDoc)
          continue;
        IndexRequest.IDs.insert(*S.IDForDoc);
      }
      Index->lookup(IndexRequest, [&](const Symbol &S) {
        if (!S.Documentation.empty())
          FetchedDocs[S.ID] = S.Documentation;
      });
      log("SigHelp: requested docs for {0} symbols from the index, got {1} "
          "symbols with non-empty docs in the response",
          IndexRequest.IDs.size(), FetchedDocs.size());
    }

    llvm::sort(
        ScoredSignatures,
        [](const ScoredSignature &L, const ScoredSignature &R) {
          // Ordering follows:
          // - Less number of parameters is better.
          // - Function is better than FunctionType which is better than
          // Function Template.
          // - High score is better.
          // - Shorter signature is better.
          // - Alphebatically smaller is better.
          if (L.Quality.NumberOfParameters != R.Quality.NumberOfParameters)
            return L.Quality.NumberOfParameters < R.Quality.NumberOfParameters;
          if (L.Quality.NumberOfOptionalParameters !=
              R.Quality.NumberOfOptionalParameters)
            return L.Quality.NumberOfOptionalParameters <
                   R.Quality.NumberOfOptionalParameters;
          if (L.Quality.Kind != R.Quality.Kind) {
            using OC = CodeCompleteConsumer::OverloadCandidate;
            switch (L.Quality.Kind) {
            case OC::CK_Function:
              return true;
            case OC::CK_FunctionType:
              return R.Quality.Kind != OC::CK_Function;
            case OC::CK_FunctionTemplate:
              return false;
            }
            llvm_unreachable("Unknown overload candidate type.");
          }
          if (L.Signature.label.size() != R.Signature.label.size())
            return L.Signature.label.size() < R.Signature.label.size();
          return L.Signature.label < R.Signature.label;
        });

    for (auto &SS : ScoredSignatures) {
      auto IndexDocIt =
          SS.IDForDoc ? FetchedDocs.find(*SS.IDForDoc) : FetchedDocs.end();
      if (IndexDocIt != FetchedDocs.end())
        SS.Signature.documentation = IndexDocIt->second;

      SigHelp.signatures.push_back(std::move(SS.Signature));
    }
  }

  GlobalCodeCompletionAllocator &getAllocator() override { return *Allocator; }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }

private:
  // FIXME(ioeric): consider moving CodeCompletionString logic here to
  // CompletionString.h.
  ScoredSignature processOverloadCandidate(const OverloadCandidate &Candidate,
                                           const CodeCompletionString &CCS,
                                           StringRef DocComment) const {
    SignatureInformation Signature;
    SignatureQualitySignals Signal;
    const char *ReturnType = nullptr;

    Signature.documentation = formatDocumentation(CCS, DocComment);
    Signal.Kind = Candidate.getKind();

    for (const auto &Chunk : CCS) {
      switch (Chunk.Kind) {
      case CodeCompletionString::CK_ResultType:
        // A piece of text that describes the type of an entity or,
        // for functions and methods, the return type.
        assert(!ReturnType && "Unexpected CK_ResultType");
        ReturnType = Chunk.Text;
        break;
      case CodeCompletionString::CK_Placeholder:
        // A string that acts as a placeholder for, e.g., a function call
        // argument.
        // Intentional fallthrough here.
      case CodeCompletionString::CK_CurrentParameter: {
        // A piece of text that describes the parameter that corresponds to
        // the code-completion location within a function call, message send,
        // macro invocation, etc.
        Signature.label += Chunk.Text;
        ParameterInformation Info;
        Info.label = Chunk.Text;
        Signature.parameters.push_back(std::move(Info));
        Signal.NumberOfParameters++;
        Signal.ContainsActiveParameter = true;
        break;
      }
      case CodeCompletionString::CK_Optional: {
        // The rest of the parameters are defaulted/optional.
        assert(Chunk.Optional &&
               "Expected the optional code completion string to be non-null.");
        Signature.label += getOptionalParameters(*Chunk.Optional,
                                                 Signature.parameters, Signal);
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
    Result.IDForDoc =
        Result.Signature.documentation.empty() && Candidate.getFunction()
            ? clangd::getSymbolID(Candidate.getFunction())
            : None;
    return Result;
  }

  SignatureHelp &SigHelp;
  std::shared_ptr<clang::GlobalCodeCompletionAllocator> Allocator;
  CodeCompletionTUInfo CCTUInfo;
  const SymbolIndex *Index;
}; // SignatureHelpCollector

struct SemaCompleteInput {
  PathRef FileName;
  const tooling::CompileCommand &Command;
  const PreambleData *Preamble;
  StringRef Contents;
  Position Pos;
  IntrusiveRefCntPtr<vfs::FileSystem> VFS;
  std::shared_ptr<PCHContainerOperations> PCHs;
};

// Invokes Sema code completion on a file.
// If \p Includes is set, it will be updated based on the compiler invocation.
bool semaCodeComplete(std::unique_ptr<CodeCompleteConsumer> Consumer,
                      const clang::CodeCompleteOptions &Options,
                      const SemaCompleteInput &Input,
                      IncludeStructure *Includes = nullptr) {
  trace::Span Tracer("Sema completion");
  std::vector<const char *> ArgStrs;
  for (const auto &S : Input.Command.CommandLine)
    ArgStrs.push_back(S.c_str());

  if (Input.VFS->setCurrentWorkingDirectory(Input.Command.Directory)) {
    log("Couldn't set working directory");
    // We run parsing anyway, our lit-tests rely on results for non-existing
    // working dirs.
  }

  IntrusiveRefCntPtr<vfs::FileSystem> VFS = Input.VFS;
  if (Input.Preamble && Input.Preamble->StatCache)
    VFS = Input.Preamble->StatCache->getConsumingFS(std::move(VFS));
  IgnoreDiagnostics DummyDiagsConsumer;
  auto CI = createInvocationFromCommandLine(
      ArgStrs,
      CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                          &DummyDiagsConsumer, false),
      VFS);
  if (!CI) {
    elog("Couldn't create CompilerInvocation");
    return false;
  }
  auto &FrontendOpts = CI->getFrontendOpts();
  FrontendOpts.DisableFree = false;
  FrontendOpts.SkipFunctionBodies = true;
  CI->getLangOpts()->CommentOpts.ParseAllComments = true;
  // Disable typo correction in Sema.
  CI->getLangOpts()->SpellChecking = false;
  // Setup code completion.
  FrontendOpts.CodeCompleteOpts = Options;
  FrontendOpts.CodeCompletionAt.FileName = Input.FileName;
  auto Offset = positionToOffset(Input.Contents, Input.Pos);
  if (!Offset) {
    elog("Code completion position was invalid {0}", Offset.takeError());
    return false;
  }
  std::tie(FrontendOpts.CodeCompletionAt.Line,
           FrontendOpts.CodeCompletionAt.Column) =
      offsetToClangLineColumn(Input.Contents, *Offset);

  std::unique_ptr<MemoryBuffer> ContentsBuffer =
      MemoryBuffer::getMemBufferCopy(Input.Contents, Input.FileName);
  // The diagnostic options must be set before creating a CompilerInstance.
  CI->getDiagnosticOpts().IgnoreWarnings = true;
  // We reuse the preamble whether it's valid or not. This is a
  // correctness/performance tradeoff: building without a preamble is slow, and
  // completion is latency-sensitive.
  // However, if we're completing *inside* the preamble section of the draft,
  // overriding the preamble will break sema completion. Fortunately we can just
  // skip all includes in this case; these completions are really simple.
  bool CompletingInPreamble =
      ComputePreambleBounds(*CI->getLangOpts(), ContentsBuffer.get(), 0).Size >
      *Offset;
  // NOTE: we must call BeginSourceFile after prepareCompilerInstance. Otherwise
  // the remapped buffers do not get freed.
  auto Clang = prepareCompilerInstance(
      std::move(CI),
      (Input.Preamble && !CompletingInPreamble) ? &Input.Preamble->Preamble
                                                : nullptr,
      std::move(ContentsBuffer), std::move(Input.PCHs), std::move(VFS),
      DummyDiagsConsumer);
  Clang->getPreprocessorOpts().SingleFileParseMode = CompletingInPreamble;
  Clang->setCodeCompletionConsumer(Consumer.release());

  SyntaxOnlyAction Action;
  if (!Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0])) {
    log("BeginSourceFile() failed when running codeComplete for {0}",
        Input.FileName);
    return false;
  }
  if (Includes)
    Clang->getPreprocessor().addPPCallbacks(
        collectIncludeStructureCallback(Clang->getSourceManager(), Includes));
  if (!Action.Execute()) {
    log("Execute() failed when running codeComplete for {0}", Input.FileName);
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
Optional<FuzzyFindRequest> speculativeFuzzyFindRequestForCompletion(
    FuzzyFindRequest CachedReq, PathRef File, StringRef Content, Position Pos) {
  auto Filter = speculateCompletionFilter(Content, Pos);
  if (!Filter) {
    elog("Failed to speculate filter text for code completion at Pos "
         "{0}:{1}: {2}",
         Pos.line, Pos.character, Filter.takeError());
    return None;
  }
  CachedReq.Query = *Filter;
  return CachedReq;
}

} // namespace

clang::CodeCompleteOptions CodeCompleteOptions::getClangCompleteOpts() const {
  clang::CodeCompleteOptions Result;
  Result.IncludeCodePatterns = EnableSnippets && IncludeCodePatterns;
  Result.IncludeMacros = IncludeMacros;
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

// Returns the most popular include header for \p Sym. If two headers are
// equally popular, prefer the shorter one. Returns empty string if \p Sym has
// no include header.
SmallVector<StringRef, 1> getRankedIncludes(const Symbol &Sym) {
  auto Includes = Sym.IncludeHeaders;
  // Sort in descending order by reference count and header length.
  llvm::sort(Includes, [](const Symbol::IncludeHeaderWithReferences &LHS,
                          const Symbol::IncludeHeaderWithReferences &RHS) {
    if (LHS.References == RHS.References)
      return LHS.IncludeHeader.size() < RHS.IncludeHeader.size();
    return LHS.References > RHS.References;
  });
  SmallVector<StringRef, 1> Headers;
  for (const auto &Include : Includes)
    Headers.push_back(Include.IncludeHeader);
  return Headers;
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
  IncludeStructure Includes; // Complete once the compiler runs.
  SpeculativeFuzzyFind *SpecFuzzyFind; // Can be nullptr.
  const CodeCompleteOptions &Opts;

  // Sema takes ownership of Recorder. Recorder is valid until Sema cleanup.
  CompletionRecorder *Recorder = nullptr;
  int NSema = 0, NIndex = 0, NBoth = 0; // Counters for logging.
  bool Incomplete = false; // Would more be available with a higher limit?
  Optional<FuzzyMatcher> Filter;        // Initialized once Sema runs.
  std::vector<std::string> QueryScopes; // Initialized once Sema runs.
  // Initialized once QueryScopes is initialized, if there are scopes.
  Optional<ScopeDistance> ScopeProximity;
  llvm::Optional<OpaqueType> PreferredType; // Initialized once Sema runs.
  // Whether to query symbols from any scope. Initialized once Sema runs.
  bool AllScopes = false;
  // Include-insertion and proximity scoring rely on the include structure.
  // This is available after Sema has run.
  Optional<IncludeInserter> Inserter;  // Available during runWithSema.
  Optional<URIDistance> FileProximity; // Initialized once Sema runs.
  /// Speculative request based on the cached request and the filter text before
  /// the cursor.
  /// Initialized right before sema run. This is only set if `SpecFuzzyFind` is
  /// set and contains a cached request.
  Optional<FuzzyFindRequest> SpecReq;

public:
  // A CodeCompleteFlow object is only useful for calling run() exactly once.
  CodeCompleteFlow(PathRef FileName, const IncludeStructure &Includes,
                   SpeculativeFuzzyFind *SpecFuzzyFind,
                   const CodeCompleteOptions &Opts)
      : FileName(FileName), Includes(Includes), SpecFuzzyFind(SpecFuzzyFind),
        Opts(Opts) {}

  CodeCompleteResult run(const SemaCompleteInput &SemaCCInput) && {
    trace::Span Tracer("CodeCompleteFlow");
    if (Opts.Index && SpecFuzzyFind && SpecFuzzyFind->CachedReq.hasValue()) {
      assert(!SpecFuzzyFind->Result.valid());
      if ((SpecReq = speculativeFuzzyFindRequestForCompletion(
               *SpecFuzzyFind->CachedReq, SemaCCInput.FileName,
               SemaCCInput.Contents, SemaCCInput.Pos)))
        SpecFuzzyFind->Result = startAsyncFuzzyFind(*Opts.Index, *SpecReq);
    }

    // We run Sema code completion first. It builds an AST and calculates:
    //   - completion results based on the AST.
    //   - partial identifier and context. We need these for the index query.
    CodeCompleteResult Output;
    auto RecorderOwner = llvm::make_unique<CompletionRecorder>(Opts, [&]() {
      assert(Recorder && "Recorder is not set");
      auto Style =
          format::getStyle(format::DefaultFormatStyle, SemaCCInput.FileName,
                           format::DefaultFallbackStyle, SemaCCInput.Contents,
                           SemaCCInput.VFS.get());
      if (!Style) {
        log("getStyle() failed for file {0}: {1}. Fallback is LLVM style.",
            SemaCCInput.FileName, Style.takeError());
        Style = format::getLLVMStyle();
      }
      // If preprocessor was run, inclusions from preprocessor callback should
      // already be added to Includes.
      Inserter.emplace(
          SemaCCInput.FileName, SemaCCInput.Contents, *Style,
          SemaCCInput.Command.Directory,
          Recorder->CCSema->getPreprocessor().getHeaderSearchInfo());
      for (const auto &Inc : Includes.MainFileIncludes)
        Inserter->addExisting(Inc);

      // Most of the cost of file proximity is in initializing the FileDistance
      // structures based on the observed includes, once per query. Conceptually
      // that happens here (though the per-URI-scheme initialization is lazy).
      // The per-result proximity scoring is (amortized) very cheap.
      FileDistanceOptions ProxOpts{}; // Use defaults.
      const auto &SM = Recorder->CCSema->getSourceManager();
      StringMap<SourceParams> ProxSources;
      for (auto &Entry : Includes.includeDepth(
               SM.getFileEntryForID(SM.getMainFileID())->getName())) {
        auto &Source = ProxSources[Entry.getKey()];
        Source.Cost = Entry.getValue() * ProxOpts.IncludeCost;
        // Symbols near our transitive includes are good, but only consider
        // things in the same directory or below it. Otherwise there can be
        // many false positives.
        if (Entry.getValue() > 0)
          Source.MaxUpTraversals = 1;
      }
      FileProximity.emplace(ProxSources, ProxOpts);

      Output = runWithSema();
      Inserter.reset(); // Make sure this doesn't out-live Clang.
      SPAN_ATTACH(Tracer, "sema_completion_kind",
                  getCompletionKindString(Recorder->CCContext.getKind()));
      log("Code complete: sema context {0}, query scopes [{1}] (AnyScope={2}), "
          "expected type {3}",
          getCompletionKindString(Recorder->CCContext.getKind()),
          join(QueryScopes.begin(), QueryScopes.end(), ","), AllScopes,
          PreferredType ? Recorder->CCContext.getPreferredType().getAsString()
                        : "<none>");
    });

    Recorder = RecorderOwner.get();

    semaCodeComplete(std::move(RecorderOwner), Opts.getClangCompleteOpts(),
                     SemaCCInput, &Includes);

    SPAN_ATTACH(Tracer, "sema_results", NSema);
    SPAN_ATTACH(Tracer, "index_results", NIndex);
    SPAN_ATTACH(Tracer, "merged_results", NBoth);
    SPAN_ATTACH(Tracer, "returned_results", int64_t(Output.Completions.size()));
    SPAN_ATTACH(Tracer, "incomplete", Output.HasMore);
    log("Code complete: {0} results from Sema, {1} from Index, "
        "{2} matched, {3} returned{4}.",
        NSema, NIndex, NBoth, Output.Completions.size(),
        Output.HasMore ? " (incomplete)" : "");
    assert(!Opts.Limit || Output.Completions.size() <= Opts.Limit);
    // We don't assert that isIncomplete means we hit a limit.
    // Indexes may choose to impose their own limits even if we don't have one.
    return Output;
  }

private:
  // This is called by run() once Sema code completion is done, but before the
  // Sema data structures are torn down. It does all the real work.
  CodeCompleteResult runWithSema() {
    const auto &CodeCompletionRange = CharSourceRange::getCharRange(
        Recorder->CCSema->getPreprocessor().getCodeCompletionTokenRange());
    Range TextEditRange;
    // When we are getting completions with an empty identifier, for example
    //    std::vector<int> asdf;
    //    asdf.^;
    // Then the range will be invalid and we will be doing insertion, use
    // current cursor position in such cases as range.
    if (CodeCompletionRange.isValid()) {
      TextEditRange = halfOpenToRange(Recorder->CCSema->getSourceManager(),
                                      CodeCompletionRange);
    } else {
      const auto &Pos = sourceLocToPosition(
          Recorder->CCSema->getSourceManager(),
          Recorder->CCSema->getPreprocessor().getCodeCompletionLoc());
      TextEditRange.start = TextEditRange.end = Pos;
    }
    Filter = FuzzyMatcher(
        Recorder->CCSema->getPreprocessor().getCodeCompletionFilter());
    std::tie(QueryScopes, AllScopes) =
        getQueryScopes(Recorder->CCContext, *Recorder->CCSema, Opts);
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
    auto Top = mergeResults(Recorder->Results, IndexResults);
    CodeCompleteResult Output;

    // Convert the results to final form, assembling the expensive strings.
    for (auto &C : Top) {
      Output.Completions.push_back(toCodeCompletion(C.first));
      Output.Completions.back().Score = C.second;
      Output.Completions.back().CompletionTokenRange = TextEditRange;
    }
    Output.HasMore = Incomplete;
    Output.Context = Recorder->CCContext.getKind();

    return Output;
  }

  SymbolSlab queryIndex() {
    trace::Span Tracer("Query index");
    SPAN_ATTACH(Tracer, "limit", int64_t(Opts.Limit));

    // Build the query.
    FuzzyFindRequest Req;
    if (Opts.Limit)
      Req.Limit = Opts.Limit;
    Req.Query = Filter->pattern();
    Req.RestrictForCodeCompletion = true;
    Req.Scopes = QueryScopes;
    Req.AnyScope = AllScopes;
    // FIXME: we should send multiple weighted paths here.
    Req.ProximityPaths.push_back(FileName);
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
    if (Opts.Index->fuzzyFind(
            Req, [&](const Symbol &Sym) { ResultsBuilder.insert(Sym); }))
      Incomplete = true;
    return std::move(ResultsBuilder).build();
  }

  // Merges Sema and Index results where possible, to form CompletionCandidates.
  // Groups overloads if desired, to form CompletionCandidate::Bundles. The
  // bundles are scored and top results are returned, best to worst.
  std::vector<ScoredBundle>
  mergeResults(const std::vector<CodeCompletionResult> &SemaResults,
               const SymbolSlab &IndexResults) {
    trace::Span Tracer("Merge and score results");
    std::vector<CompletionCandidate::Bundle> Bundles;
    DenseMap<size_t, size_t> BundleLookup;
    auto AddToBundles = [&](const CodeCompletionResult *SemaResult,
                            const Symbol *IndexResult) {
      CompletionCandidate C;
      C.SemaResult = SemaResult;
      C.IndexResult = IndexResult;
      if (C.IndexResult)
        C.RankedIncludeHeaders = getRankedIncludes(*C.IndexResult);
      C.Name = IndexResult ? IndexResult->Name : Recorder->getName(*SemaResult);
      if (auto OverloadSet = Opts.BundleOverloads ? C.overloadSet() : 0) {
        auto Ret = BundleLookup.try_emplace(OverloadSet, Bundles.size());
        if (Ret.second)
          Bundles.emplace_back();
        Bundles[Ret.first->second].push_back(std::move(C));
      } else {
        Bundles.emplace_back();
        Bundles.back().push_back(std::move(C));
      }
    };
    DenseSet<const Symbol *> UsedIndexResults;
    auto CorrespondingIndexResult =
        [&](const CodeCompletionResult &SemaResult) -> const Symbol * {
      if (auto SymID =
              getSymbolID(SemaResult, Recorder->CCSema->getSourceManager())) {
        auto I = IndexResults.find(*SymID);
        if (I != IndexResults.end()) {
          UsedIndexResults.insert(&*I);
          return &*I;
        }
      }
      return nullptr;
    };
    // Emit all Sema results, merging them with Index results if possible.
    for (auto &SemaResult : Recorder->Results)
      AddToBundles(&SemaResult, CorrespondingIndexResult(SemaResult));
    // Now emit any Index-only results.
    for (const auto &IndexResult : IndexResults) {
      if (UsedIndexResults.count(&IndexResult))
        continue;
      AddToBundles(/*SemaResult=*/nullptr, &IndexResult);
    }
    // We only keep the best N results at any time, in "native" format.
    TopN<ScoredBundle, ScoredBundleGreater> Top(
        Opts.Limit == 0 ? std::numeric_limits<size_t>::max() : Opts.Limit);
    for (auto &Bundle : Bundles)
      addCandidate(Top, std::move(Bundle));
    return std::move(Top).items();
  }

  Optional<float> fuzzyScore(const CompletionCandidate &C) {
    // Macros can be very spammy, so we only support prefix completion.
    // We won't end up with underfull index results, as macros are sema-only.
    if (C.SemaResult && C.SemaResult->Kind == CodeCompletionResult::RK_Macro &&
        !C.Name.startswith_lower(Filter->pattern()))
      return None;
    return Filter->match(C.Name);
  }

  // Scores a candidate and adds it to the TopN structure.
  void addCandidate(TopN<ScoredBundle, ScoredBundleGreater> &Candidates,
                    CompletionCandidate::Bundle Bundle) {
    SymbolQualitySignals Quality;
    SymbolRelevanceSignals Relevance;
    Relevance.Context = Recorder->CCContext.getKind();
    Relevance.Query = SymbolRelevanceSignals::CodeComplete;
    Relevance.FileProximityMatch = FileProximity.getPointer();
    if (ScopeProximity)
      Relevance.ScopeProximityMatch = ScopeProximity.getPointer();
    if (PreferredType)
      Relevance.HadContextType = true;

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
    }

    CodeCompletion::Scores Scores;
    Scores.Quality = Quality.evaluate();
    Scores.Relevance = Relevance.evaluate();
    Scores.Total = evaluateSymbolAndRelevance(Scores.Quality, Scores.Relevance);
    // NameMatch is in fact a multiplier on total score, so rescoring is sound.
    Scores.ExcludingName = Relevance.NameMatch
                               ? Scores.Total / Relevance.NameMatch
                               : Scores.Quality;

    dlog("CodeComplete: {0} ({1}) = {2}\n{3}{4}\n", First.Name,
         to_string(Origin), Scores.Total, to_string(Quality),
         to_string(Relevance));

    NSema += bool(Origin & SymbolOrigin::AST);
    NIndex += FromIndex;
    NBoth += bool(Origin & SymbolOrigin::AST) && FromIndex;
    if (Candidates.push({std::move(Bundle), Scores}))
      Incomplete = true;
  }

  CodeCompletion toCodeCompletion(const CompletionCandidate::Bundle &Bundle) {
    Optional<CodeCompletionBuilder> Builder;
    for (const auto &Item : Bundle) {
      CodeCompletionString *SemaCCS =
          Item.SemaResult ? Recorder->codeCompletionString(*Item.SemaResult)
                          : nullptr;
      if (!Builder)
        Builder.emplace(Recorder->CCSema->getASTContext(), Item, SemaCCS,
                        QueryScopes, *Inserter, FileName,
                        Recorder->CCContext.getKind(), Opts);
      else
        Builder->add(Item, SemaCCS);
    }
    return Builder->build();
  }
};

Expected<StringRef> speculateCompletionFilter(StringRef Content, Position Pos) {
  auto Offset = positionToOffset(Content, Pos);
  if (!Offset)
    return make_error<StringError>(
        "Failed to convert position to offset in content.",
        inconvertibleErrorCode());
  if (*Offset == 0)
    return "";

  // Start from the character before the cursor.
  int St = *Offset - 1;
  // FIXME(ioeric): consider UTF characters?
  auto IsValidIdentifierChar = [](char c) {
    return ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || (c == '_'));
  };
  size_t Len = 0;
  for (; (St >= 0) && IsValidIdentifierChar(Content[St]); --St, ++Len) {
  }
  if (Len > 0)
    St++; // Shift to the first valid character.
  return Content.substr(St, Len);
}

CodeCompleteResult
codeComplete(PathRef FileName, const tooling::CompileCommand &Command,
             const PreambleData *Preamble, StringRef Contents, Position Pos,
             IntrusiveRefCntPtr<vfs::FileSystem> VFS,
             std::shared_ptr<PCHContainerOperations> PCHs,
             CodeCompleteOptions Opts, SpeculativeFuzzyFind *SpecFuzzyFind) {
  return CodeCompleteFlow(FileName,
                          Preamble ? Preamble->Includes : IncludeStructure(),
                          SpecFuzzyFind, Opts)
      .run({FileName, Command, Preamble, Contents, Pos, VFS, PCHs});
}

SignatureHelp signatureHelp(PathRef FileName,
                            const tooling::CompileCommand &Command,
                            const PreambleData *Preamble, StringRef Contents,
                            Position Pos,
                            IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                            std::shared_ptr<PCHContainerOperations> PCHs,
                            const SymbolIndex *Index) {
  SignatureHelp Result;
  clang::CodeCompleteOptions Options;
  Options.IncludeGlobals = false;
  Options.IncludeMacros = false;
  Options.IncludeCodePatterns = false;
  Options.IncludeBriefComments = false;
  IncludeStructure PreambleInclusions; // Unused for signatureHelp
  semaCodeComplete(
      llvm::make_unique<SignatureHelpCollector>(Options, Index, Result),
      Options,
      {FileName, Command, Preamble, Contents, Pos, std::move(VFS),
       std::move(PCHs)});
  return Result;
}

bool isIndexedForCodeCompletion(const NamedDecl &ND, ASTContext &ASTCtx) {
  using namespace clang::ast_matchers;
  auto InTopLevelScope = hasDeclContext(
      anyOf(namespaceDecl(), translationUnitDecl(), linkageSpecDecl()));
  return !match(decl(anyOf(InTopLevelScope,
                           hasDeclContext(
                               enumDecl(InTopLevelScope, unless(isScoped()))))),
                ND, ASTCtx)
              .empty();
}

CompletionItem CodeCompletion::render(const CodeCompleteOptions &Opts) const {
  CompletionItem LSP;
  const auto *InsertInclude = Includes.empty() ? nullptr : &Includes[0];
  LSP.label = ((InsertInclude && InsertInclude->Insertion)
                   ? Opts.IncludeIndicator.Insert
                   : Opts.IncludeIndicator.NoInsert) +
              (Opts.ShowOrigins ? "[" + to_string(Origin) + "]" : "") +
              RequiredQualifier + Name + Signature;

  LSP.kind = Kind;
  LSP.detail =
      BundleSize > 1 ? formatv("[{0} overloads]", BundleSize) : ReturnType;
  LSP.deprecated = Deprecated;
  if (InsertInclude)
    LSP.detail += "\n" + InsertInclude->Header;
  LSP.documentation = Documentation;
  LSP.sortText = sortText(Score.Total, Name);
  LSP.filterText = Name;
  LSP.textEdit = {CompletionTokenRange, RequiredQualifier + Name};
  // Merge continuous additionalTextEdits into main edit. The main motivation
  // behind this is to help LSP clients, it seems most of them are confused when
  // they are provided with additionalTextEdits that are consecutive to main
  // edit.
  // Note that we store additional text edits from back to front in a line. That
  // is mainly to help LSP clients again, so that changes do not effect each
  // other.
  for (const auto &FixIt : FixIts) {
    if (IsRangeConsecutive(FixIt.range, LSP.textEdit->range)) {
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
  LSP.insertTextFormat = Opts.EnableSnippets ? InsertTextFormat::Snippet
                                             : InsertTextFormat::PlainText;
  if (InsertInclude && InsertInclude->Insertion)
    LSP.additionalTextEdits.push_back(*InsertInclude->Insertion);

  return LSP;
}

raw_ostream &operator<<(raw_ostream &OS, const CodeCompletion &C) {
  // For now just lean on CompletionItem.
  return OS << C.render(CodeCompleteOptions());
}

raw_ostream &operator<<(raw_ostream &OS, const CodeCompleteResult &R) {
  OS << "CodeCompleteResult: " << R.Completions.size() << (R.HasMore ? "+" : "")
     << " (" << getCompletionKindString(R.Context) << ")"
     << " items:\n";
  for (const auto &C : R.Completions)
    OS << C << "\n";
  return OS;
}

} // namespace clangd
} // namespace clang
