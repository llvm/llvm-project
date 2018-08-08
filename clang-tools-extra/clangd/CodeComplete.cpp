//===--- CodeComplete.cpp ---------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
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
//===---------------------------------------------------------------------===//

#include "CodeComplete.h"
#include "AST.h"
#include "CodeCompletionStrings.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "FileDistance.h"
#include "FuzzyMatch.h"
#include "Headers.h"
#include "Logger.h"
#include "Quality.h"
#include "SourceCode.h"
#include "Trace.h"
#include "URI.h"
#include "index/Index.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/Sema.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ScopedPrinter.h"
#include <queue>

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
                     const NamedDecl *Decl) {
  if (Decl)
    return toCompletionItemKind(index::getSymbolInfo(Decl).Kind);
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
std::string
getOptionalParameters(const CodeCompletionString &CCS,
                      std::vector<ParameterInformation> &Parameters) {
  std::string Result;
  for (const auto &Chunk : CCS) {
    switch (Chunk.Kind) {
    case CodeCompletionString::CK_Optional:
      assert(Chunk.Optional &&
             "Expected the optional code completion string to be non-null.");
      Result += getOptionalParameters(*Chunk.Optional, Parameters);
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
static llvm::Expected<HeaderFile> toHeaderFile(StringRef Header,
                                               llvm::StringRef HintPath) {
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
  llvm::StringRef Name; // Used for filtering and sorting.
  // We may have a result from Sema, from the index, or both.
  const CodeCompletionResult *SemaResult = nullptr;
  const Symbol *IndexResult = nullptr;

  // Returns a token identifying the overload set this is part of.
  // 0 indicates it's not part of any overload set.
  size_t overloadSet() const {
    SmallString<256> Scratch;
    if (IndexResult) {
      switch (IndexResult->SymInfo.Kind) {
      case index::SymbolKind::ClassMethod:
      case index::SymbolKind::InstanceMethod:
      case index::SymbolKind::StaticMethod:
        assert(false && "Don't expect members from index in code completion");
        // fall through
      case index::SymbolKind::Function:
        // We can't group overloads together that need different #includes.
        // This could break #include insertion.
        return hash_combine(
            (IndexResult->Scope + IndexResult->Name).toStringRef(Scratch),
            headerToInsertIfNotPresent().getValueOr(""));
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
      llvm::raw_svector_ostream OS(Scratch);
      D->printQualifiedName(OS);
    }
    return hash_combine(Scratch, headerToInsertIfNotPresent().getValueOr(""));
  }

  llvm::Optional<llvm::StringRef> headerToInsertIfNotPresent() const {
    if (!IndexResult || !IndexResult->Detail ||
        IndexResult->Detail->IncludeHeader.empty())
      return llvm::None;
    if (SemaResult && SemaResult->Declaration) {
      // Avoid inserting new #include if the declaration is found in the current
      // file e.g. the symbol is forward declared.
      auto &SM = SemaResult->Declaration->getASTContext().getSourceManager();
      for (const Decl *RD : SemaResult->Declaration->redecls())
        if (SM.isInMainFile(SM.getExpansionLoc(RD->getLocStart())))
          return llvm::None;
    }
    return IndexResult->Detail->IncludeHeader;
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
  CodeCompletionBuilder(ASTContext &ASTCtx, const CompletionCandidate &C,
                        CodeCompletionString *SemaCCS,
                        const IncludeInserter &Includes, StringRef FileName,
                        const CodeCompleteOptions &Opts)
      : ASTCtx(ASTCtx), ExtractDocumentation(Opts.IncludeComments) {
    add(C, SemaCCS);
    if (C.SemaResult) {
      Completion.Origin |= SymbolOrigin::AST;
      Completion.Name = llvm::StringRef(SemaCCS->getTypedText());
      if (Completion.Scope.empty()) {
        if ((C.SemaResult->Kind == CodeCompletionResult::RK_Declaration) ||
            (C.SemaResult->Kind == CodeCompletionResult::RK_Pattern))
          if (const auto *D = C.SemaResult->getDeclaration())
            if (const auto *ND = llvm::dyn_cast<NamedDecl>(D))
              Completion.Scope =
                  splitQualifiedName(printQualifiedName(*ND)).first;
      }
      Completion.Kind =
          toCompletionItemKind(C.SemaResult->Kind, C.SemaResult->Declaration);
      for (const auto &FixIt : C.SemaResult->FixIts) {
        Completion.FixIts.push_back(
            toTextEdit(FixIt, ASTCtx.getSourceManager(), ASTCtx.getLangOpts()));
      }
    }
    if (C.IndexResult) {
      Completion.Origin |= C.IndexResult->Origin;
      if (Completion.Scope.empty())
        Completion.Scope = C.IndexResult->Scope;
      if (Completion.Kind == CompletionItemKind::Missing)
        Completion.Kind = toCompletionItemKind(C.IndexResult->SymInfo.Kind);
      if (Completion.Name.empty())
        Completion.Name = C.IndexResult->Name;
    }
    if (auto Inserted = C.headerToInsertIfNotPresent()) {
      // Turn absolute path into a literal string that can be #included.
      auto Include = [&]() -> Expected<std::pair<std::string, bool>> {
        auto ResolvedDeclaring =
            toHeaderFile(C.IndexResult->CanonicalDeclaration.FileURI, FileName);
        if (!ResolvedDeclaring)
          return ResolvedDeclaring.takeError();
        auto ResolvedInserted = toHeaderFile(*Inserted, FileName);
        if (!ResolvedInserted)
          return ResolvedInserted.takeError();
        return std::make_pair(Includes.calculateIncludePath(*ResolvedDeclaring,
                                                            *ResolvedInserted),
                              Includes.shouldInsertInclude(*ResolvedDeclaring,
                                                           *ResolvedInserted));
      }();
      if (Include) {
        Completion.Header = Include->first;
        if (Include->second)
          Completion.HeaderInsertion = Includes.insert(Include->first);
      } else
        log("Failed to generate include insertion edits for adding header "
            "(FileURI='{0}', IncludeHeader='{1}') into {2}",
            C.IndexResult->CanonicalDeclaration.FileURI,
            C.IndexResult->Detail->IncludeHeader, FileName);
    }
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
      if (auto *D = C.IndexResult->Detail)
        S.ReturnType = D->ReturnType;
    }
    if (ExtractDocumentation && Completion.Documentation.empty()) {
      if (C.IndexResult && C.IndexResult->Detail)
        Completion.Documentation = C.IndexResult->Detail->Documentation;
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

  std::string summarizeReturnType() const {
    if (auto *RT = onlyValue<&BundledEntry::ReturnType>())
      return *RT;
    return "";
  }

  std::string summarizeSnippet() const {
    if (auto *Snippet = onlyValue<&BundledEntry::SnippetSuffix>())
      return *Snippet;
    // All bundles are function calls.
    return "(${0})";
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
};

// Determine the symbol ID for a Sema code completion result, if possible.
llvm::Optional<SymbolID> getSymbolID(const CodeCompletionResult &R) {
  switch (R.Kind) {
  case CodeCompletionResult::RK_Declaration:
  case CodeCompletionResult::RK_Pattern: {
    return clang::clangd::getSymbolID(R.Declaration);
  }
  case CodeCompletionResult::RK_Macro:
    // FIXME: Macros do have USRs, but the CCR doesn't contain enough info.
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
  llvm::Optional<std::string> UnresolvedQualifier;

  // Construct scopes being queried in indexes.
  // This method format the scopes to match the index request representation.
  std::vector<std::string> scopesForIndexQuery() {
    std::vector<std::string> Results;
    for (llvm::StringRef AS : AccessibleScopes) {
      Results.push_back(AS);
      if (UnresolvedQualifier)
        Results.back() += *UnresolvedQualifier;
    }
    return Results;
  }
};

// Get all scopes that will be queried in indexes.
std::vector<std::string> getQueryScopes(CodeCompletionContext &CCContext,
                                        const SourceManager &SM) {
  auto GetAllAccessibleScopes = [](CodeCompletionContext &CCContext) {
    SpecifiedScope Info;
    for (auto *Context : CCContext.getVisitedContexts()) {
      if (isa<TranslationUnitDecl>(Context))
        Info.AccessibleScopes.push_back(""); // global namespace
      else if (const auto *NS = dyn_cast<NamespaceDecl>(Context))
        Info.AccessibleScopes.push_back(NS->getQualifiedNameAsString() + "::");
    }
    return Info;
  };

  auto SS = CCContext.getCXXScopeSpecifier();

  // Unqualified completion (e.g. "vec^").
  if (!SS) {
    // FIXME: Once we can insert namespace qualifiers and use the in-scope
    //        namespaces for scoring, search in all namespaces.
    // FIXME: Capture scopes and use for scoring, for example,
    //        "using namespace std; namespace foo {v^}" =>
    //        foo::value > std::vector > boost::variant
    return GetAllAccessibleScopes(CCContext).scopesForIndexQuery();
  }

  // Qualified completion ("std::vec^"), we have two cases depending on whether
  // the qualifier can be resolved by Sema.
  if ((*SS)->isValid()) { // Resolved qualifier.
    return GetAllAccessibleScopes(CCContext).scopesForIndexQuery();
  }

  // Unresolved qualifier.
  // FIXME: When Sema can resolve part of a scope chain (e.g.
  // "known::unknown::id"), we should expand the known part ("known::") rather
  // than treating the whole thing as unknown.
  SpecifiedScope Info;
  Info.AccessibleScopes.push_back(""); // global namespace

  Info.UnresolvedQualifier =
      Lexer::getSourceText(CharSourceRange::getCharRange((*SS)->getRange()), SM,
                           clang::LangOptions())
          .ltrim("::");
  // Sema excludes the trailing "::".
  if (!Info.UnresolvedQualifier->empty())
    *Info.UnresolvedQualifier += "::";

  return Info.scopesForIndexQuery();
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
  case CodeCompletionContext::CCC_Name: // FIXME: why does ns::^ give this?
  case CodeCompletionContext::CCC_PotentiallyQualifiedName:
  case CodeCompletionContext::CCC_ParenthesizedExpression:
  case CodeCompletionContext::CCC_ObjCInterfaceName:
  case CodeCompletionContext::CCC_ObjCCategoryName:
    return true;
  case CodeCompletionContext::CCC_Other: // Be conservative.
  case CodeCompletionContext::CCC_OtherWithMacros:
  case CodeCompletionContext::CCC_DotMemberAccess:
  case CodeCompletionContext::CCC_ArrowMemberAccess:
  case CodeCompletionContext::CCC_ObjCPropertyAccess:
  case CodeCompletionContext::CCC_MacroName:
  case CodeCompletionContext::CCC_MacroNameUse:
  case CodeCompletionContext::CCC_PreprocessorExpression:
  case CodeCompletionContext::CCC_PreprocessorDirective:
  case CodeCompletionContext::CCC_NaturalLanguage:
  case CodeCompletionContext::CCC_SelectorName:
  case CodeCompletionContext::CCC_TypeQualifiers:
  case CodeCompletionContext::CCC_ObjCInstanceMessage:
  case CodeCompletionContext::CCC_ObjCClassMessage:
  case CodeCompletionContext::CCC_Recovery:
    return false;
  }
  llvm_unreachable("unknown code completion context");
}

// Some member calls are blacklisted because they're so rarely useful.
static bool isBlacklistedMember(const NamedDecl &D) {
  // Destructor completion is rarely useful, and works inconsistently.
  // (s.^ completes ~string, but s.~st^ is an error).
  if (D.getKind() == Decl::CXXDestructor)
    return true;
  // Injected name may be useful for A::foo(), but who writes A::A::foo()?
  if (auto *R = dyn_cast_or_null<RecordDecl>(&D))
    if (R->isInjectedClassName())
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
      // Drop hidden items which cannot be found by lookup after completion.
      // Exception: some items can be named by using a qualifier.
      if (Result.Hidden && (!Result.Qualifier || Result.QualifierIsInformative))
        continue;
      if (!Opts.IncludeIneligibleResults &&
          (Result.Availability == CXAvailability_NotAvailable ||
           Result.Availability == CXAvailability_NotAccessible))
        continue;
      if (Result.Declaration &&
          !Context.getBaseType().isNull() // is this a member-access context?
          && isBlacklistedMember(*Result.Declaration))
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
  llvm::unique_function<void()> ResultsCallback;
};

class SignatureHelpCollector final : public CodeCompleteConsumer {

public:
  SignatureHelpCollector(const clang::CodeCompleteOptions &CodeCompleteOpts,
                         SignatureHelp &SigHelp)
      : CodeCompleteConsumer(CodeCompleteOpts, /*OutputIsBinary=*/false),
        SigHelp(SigHelp),
        Allocator(std::make_shared<clang::GlobalCodeCompletionAllocator>()),
        CCTUInfo(Allocator) {}

  void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                 OverloadCandidate *Candidates,
                                 unsigned NumCandidates) override {
    SigHelp.signatures.reserve(NumCandidates);
    // FIXME(rwols): How can we determine the "active overload candidate"?
    // Right now the overloaded candidates seem to be provided in a "best fit"
    // order, so I'm not too worried about this.
    SigHelp.activeSignature = 0;
    assert(CurrentArg <= (unsigned)std::numeric_limits<int>::max() &&
           "too many arguments");
    SigHelp.activeParameter = static_cast<int>(CurrentArg);
    for (unsigned I = 0; I < NumCandidates; ++I) {
      const auto &Candidate = Candidates[I];
      const auto *CCS = Candidate.CreateSignatureString(
          CurrentArg, S, *Allocator, CCTUInfo, true);
      assert(CCS && "Expected the CodeCompletionString to be non-null");
      // FIXME: for headers, we need to get a comment from the index.
      SigHelp.signatures.push_back(processOverloadCandidate(
          Candidate, *CCS,
          getParameterDocComment(S.getASTContext(), Candidate, CurrentArg,
                                 /*CommentsFromHeaders=*/false)));
    }
  }

  GlobalCodeCompletionAllocator &getAllocator() override { return *Allocator; }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }

private:
  // FIXME(ioeric): consider moving CodeCompletionString logic here to
  // CompletionString.h.
  SignatureInformation
  processOverloadCandidate(const OverloadCandidate &Candidate,
                           const CodeCompletionString &CCS,
                           llvm::StringRef DocComment) const {
    SignatureInformation Result;
    const char *ReturnType = nullptr;

    Result.documentation = formatDocumentation(CCS, DocComment);

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
        Result.label += Chunk.Text;
        ParameterInformation Info;
        Info.label = Chunk.Text;
        Result.parameters.push_back(std::move(Info));
        break;
      }
      case CodeCompletionString::CK_Optional: {
        // The rest of the parameters are defaulted/optional.
        assert(Chunk.Optional &&
               "Expected the optional code completion string to be non-null.");
        Result.label +=
            getOptionalParameters(*Chunk.Optional, Result.parameters);
        break;
      }
      case CodeCompletionString::CK_VerticalSpace:
        break;
      default:
        Result.label += Chunk.Text;
        break;
      }
    }
    if (ReturnType) {
      Result.label += " -> ";
      Result.label += ReturnType;
    }
    return Result;
  }

  SignatureHelp &SigHelp;
  std::shared_ptr<clang::GlobalCodeCompletionAllocator> Allocator;
  CodeCompletionTUInfo CCTUInfo;

}; // SignatureHelpCollector

struct SemaCompleteInput {
  PathRef FileName;
  const tooling::CompileCommand &Command;
  PrecompiledPreamble const *Preamble;
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

  IgnoreDiagnostics DummyDiagsConsumer;
  auto CI = createInvocationFromCommandLine(
      ArgStrs,
      CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                          &DummyDiagsConsumer, false),
      Input.VFS);
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

  std::unique_ptr<llvm::MemoryBuffer> ContentsBuffer =
      llvm::MemoryBuffer::getMemBufferCopy(Input.Contents, Input.FileName);
  // The diagnostic options must be set before creating a CompilerInstance.
  CI->getDiagnosticOpts().IgnoreWarnings = true;
  // We reuse the preamble whether it's valid or not. This is a
  // correctness/performance tradeoff: building without a preamble is slow, and
  // completion is latency-sensitive.
  // NOTE: we must call BeginSourceFile after prepareCompilerInstance. Otherwise
  // the remapped buffers do not get freed.
  auto Clang = prepareCompilerInstance(
      std::move(CI), Input.Preamble, std::move(ContentsBuffer),
      std::move(Input.PCHs), std::move(Input.VFS), DummyDiagsConsumer);
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
  const CodeCompleteOptions &Opts;
  // Sema takes ownership of Recorder. Recorder is valid until Sema cleanup.
  CompletionRecorder *Recorder = nullptr;
  int NSema = 0, NIndex = 0, NBoth = 0; // Counters for logging.
  bool Incomplete = false; // Would more be available with a higher limit?
  llvm::Optional<FuzzyMatcher> Filter;       // Initialized once Sema runs.
  std::vector<std::string> QueryScopes;      // Initialized once Sema runs.
  // Include-insertion and proximity scoring rely on the include structure.
  // This is available after Sema has run.
  llvm::Optional<IncludeInserter> Inserter;  // Available during runWithSema.
  llvm::Optional<URIDistance> FileProximity; // Initialized once Sema runs.

public:
  // A CodeCompleteFlow object is only useful for calling run() exactly once.
  CodeCompleteFlow(PathRef FileName, const IncludeStructure &Includes,
                   const CodeCompleteOptions &Opts)
      : FileName(FileName), Includes(Includes), Opts(Opts) {}

  CodeCompleteResult run(const SemaCompleteInput &SemaCCInput) && {
    trace::Span Tracer("CodeCompleteFlow");

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
      llvm::StringMap<SourceParams> ProxSources;
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
      log("Code complete: sema context {0}, query scopes [{1}]",
          getCompletionKindString(Recorder->CCContext.getKind()),
          llvm::join(QueryScopes.begin(), QueryScopes.end(), ","));
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
    Filter = FuzzyMatcher(
        Recorder->CCSema->getPreprocessor().getCodeCompletionFilter());
    QueryScopes = getQueryScopes(Recorder->CCContext,
                                 Recorder->CCSema->getSourceManager());
    // Sema provides the needed context to query the index.
    // FIXME: in addition to querying for extra/overlapping symbols, we should
    //        explicitly request symbols corresponding to Sema results.
    //        We can use their signals even if the index can't suggest them.
    // We must copy index results to preserve them, but there are at most Limit.
    auto IndexResults = (Opts.Index && allowIndex(Recorder->CCContext))
                            ? queryIndex()
                            : SymbolSlab();
    // Merge Sema and Index results, score them, and pick the winners.
    auto Top = mergeResults(Recorder->Results, IndexResults);
    // Convert the results to final form, assembling the expensive strings.
    CodeCompleteResult Output;
    for (auto &C : Top) {
      Output.Completions.push_back(toCodeCompletion(C.first));
      Output.Completions.back().Score = C.second;
    }
    Output.HasMore = Incomplete;
    Output.Context = Recorder->CCContext.getKind();
    return Output;
  }

  SymbolSlab queryIndex() {
    trace::Span Tracer("Query index");
    SPAN_ATTACH(Tracer, "limit", int64_t(Opts.Limit));

    SymbolSlab::Builder ResultsBuilder;
    // Build the query.
    FuzzyFindRequest Req;
    if (Opts.Limit)
      Req.MaxCandidateCount = Opts.Limit;
    Req.Query = Filter->pattern();
    Req.RestrictForCodeCompletion = true;
    Req.Scopes = QueryScopes;
    // FIXME: we should send multiple weighted paths here.
    Req.ProximityPaths.push_back(FileName);
    vlog("Code complete: fuzzyFind(\"{0}\", scopes=[{1}])", Req.Query,
         llvm::join(Req.Scopes.begin(), Req.Scopes.end(), ","));
    // Run the query against the index.
    if (Opts.Index->fuzzyFind(
            Req, [&](const Symbol &Sym) { ResultsBuilder.insert(Sym); }))
      Incomplete = true;
    return std::move(ResultsBuilder).build();
  }

  // Merges Sema and Index results where possible, to form CompletionCandidates.
  // Groups overloads if desired, to form CompletionCandidate::Bundles.
  // The bundles are scored and top results are returned, best to worst.
  std::vector<ScoredBundle>
      mergeResults(const std::vector<CodeCompletionResult> &SemaResults,
                   const SymbolSlab &IndexResults) {
    trace::Span Tracer("Merge and score results");
    std::vector<CompletionCandidate::Bundle> Bundles;
    llvm::DenseMap<size_t, size_t> BundleLookup;
    auto AddToBundles = [&](const CodeCompletionResult *SemaResult,
                            const Symbol *IndexResult) {
      CompletionCandidate C;
      C.SemaResult = SemaResult;
      C.IndexResult = IndexResult;
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
    llvm::DenseSet<const Symbol *> UsedIndexResults;
    auto CorrespondingIndexResult =
        [&](const CodeCompletionResult &SemaResult) -> const Symbol * {
      if (auto SymID = getSymbolID(SemaResult)) {
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
      }
      if (Candidate.SemaResult) {
        Quality.merge(*Candidate.SemaResult);
        Relevance.merge(*Candidate.SemaResult);
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
         llvm::to_string(Origin), Scores.Total, llvm::to_string(Quality),
         llvm::to_string(Relevance));

    NSema += bool(Origin & SymbolOrigin::AST);
    NIndex += FromIndex;
    NBoth += bool(Origin & SymbolOrigin::AST) && FromIndex;
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
        Builder.emplace(Recorder->CCSema->getASTContext(), Item, SemaCCS,
                        *Inserter, FileName, Opts);
      else
        Builder->add(Item, SemaCCS);
    }
    return Builder->build();
  }
};

CodeCompleteResult codeComplete(PathRef FileName,
                                const tooling::CompileCommand &Command,
                                PrecompiledPreamble const *Preamble,
                                const IncludeStructure &PreambleInclusions,
                                StringRef Contents, Position Pos,
                                IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                                std::shared_ptr<PCHContainerOperations> PCHs,
                                CodeCompleteOptions Opts) {
  return CodeCompleteFlow(FileName, PreambleInclusions, Opts)
      .run({FileName, Command, Preamble, Contents, Pos, VFS, PCHs});
}

SignatureHelp signatureHelp(PathRef FileName,
                            const tooling::CompileCommand &Command,
                            PrecompiledPreamble const *Preamble,
                            StringRef Contents, Position Pos,
                            IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                            std::shared_ptr<PCHContainerOperations> PCHs) {
  SignatureHelp Result;
  clang::CodeCompleteOptions Options;
  Options.IncludeGlobals = false;
  Options.IncludeMacros = false;
  Options.IncludeCodePatterns = false;
  Options.IncludeBriefComments = false;
  IncludeStructure PreambleInclusions; // Unused for signatureHelp
  semaCodeComplete(llvm::make_unique<SignatureHelpCollector>(Options, Result),
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
  LSP.label = (HeaderInsertion ? Opts.IncludeIndicator.Insert
                               : Opts.IncludeIndicator.NoInsert) +
              (Opts.ShowOrigins ? "[" + llvm::to_string(Origin) + "]" : "") +
              RequiredQualifier + Name + Signature;

  LSP.kind = Kind;
  LSP.detail = BundleSize > 1 ? llvm::formatv("[{0} overloads]", BundleSize)
                              : ReturnType;
  if (!Header.empty())
    LSP.detail += "\n" + Header;
  LSP.documentation = Documentation;
  LSP.sortText = sortText(Score.Total, Name);
  LSP.filterText = Name;
  // FIXME(kadircet): Use LSP.textEdit instead of insertText, because it causes
  // undesired behaviours. Like completing "this.^" into "this-push_back".
  LSP.insertText = RequiredQualifier + Name;
  if (Opts.EnableSnippets)
    LSP.insertText += SnippetSuffix;
  LSP.insertTextFormat = Opts.EnableSnippets ? InsertTextFormat::Snippet
                                             : InsertTextFormat::PlainText;
  LSP.additionalTextEdits.reserve(FixIts.size() + (HeaderInsertion ? 1 : 0));
  for (const auto &FixIt : FixIts)
    LSP.additionalTextEdits.push_back(FixIt);
  if (HeaderInsertion)
    LSP.additionalTextEdits.push_back(*HeaderInsertion);
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
