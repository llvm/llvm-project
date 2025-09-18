//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UpgradeGoogletestCaseCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::google {

static const llvm::StringRef RenameCaseToSuiteMessage =
    "Google Test APIs named with 'case' are deprecated; use equivalent APIs "
    "named with 'suite'";

static std::optional<llvm::StringRef>
getNewMacroName(llvm::StringRef MacroName) {
  std::pair<llvm::StringRef, llvm::StringRef> ReplacementMap[] = {
      {"TYPED_TEST_CASE", "TYPED_TEST_SUITE"},
      {"TYPED_TEST_CASE_P", "TYPED_TEST_SUITE_P"},
      {"REGISTER_TYPED_TEST_CASE_P", "REGISTER_TYPED_TEST_SUITE_P"},
      {"INSTANTIATE_TYPED_TEST_CASE_P", "INSTANTIATE_TYPED_TEST_SUITE_P"},
      {"INSTANTIATE_TEST_CASE_P", "INSTANTIATE_TEST_SUITE_P"},
  };

  for (auto &Mapping : ReplacementMap) {
    if (MacroName == Mapping.first)
      return Mapping.second;
  }

  return std::nullopt;
}

namespace {

class UpgradeGoogletestCasePPCallback : public PPCallbacks {
public:
  UpgradeGoogletestCasePPCallback(UpgradeGoogletestCaseCheck *Check,
                                  Preprocessor *PP)
      : Check(Check), PP(PP) {}

  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *) override {
    macroUsed(MacroNameTok, MD, Range.getBegin(), CheckAction::Rename);
  }

  void MacroUndefined(const Token &MacroNameTok, const MacroDefinition &MD,
                      const MacroDirective *Undef) override {
    if (Undef != nullptr)
      macroUsed(MacroNameTok, MD, Undef->getLocation(), CheckAction::Warn);
  }

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    if (!ReplacementFound && MD != nullptr) {
      // We check if the newly defined macro is one of the target replacements.
      // This ensures that the check creates warnings only if it is including a
      // recent enough version of Google Test.
      llvm::StringRef FileName = PP->getSourceManager().getFilename(
          MD->getMacroInfo()->getDefinitionLoc());
      ReplacementFound = FileName.ends_with("gtest/gtest-typed-test.h") &&
                         PP->getSpelling(MacroNameTok) == "TYPED_TEST_SUITE";
    }
  }

  void Defined(const Token &MacroNameTok, const MacroDefinition &MD,
               SourceRange Range) override {
    macroUsed(MacroNameTok, MD, Range.getBegin(), CheckAction::Warn);
  }

  void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
             const MacroDefinition &MD) override {
    macroUsed(MacroNameTok, MD, Loc, CheckAction::Warn);
  }

  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override {
    macroUsed(MacroNameTok, MD, Loc, CheckAction::Warn);
  }

private:
  enum class CheckAction { Warn, Rename };

  void macroUsed(const clang::Token &MacroNameTok, const MacroDefinition &MD,
                 SourceLocation Loc, CheckAction Action) {
    if (!ReplacementFound)
      return;

    std::string Name = PP->getSpelling(MacroNameTok);

    std::optional<llvm::StringRef> Replacement = getNewMacroName(Name);
    if (!Replacement)
      return;

    llvm::StringRef FileName = PP->getSourceManager().getFilename(
        MD.getMacroInfo()->getDefinitionLoc());
    if (!FileName.ends_with("gtest/gtest-typed-test.h"))
      return;

    DiagnosticBuilder Diag = Check->diag(Loc, RenameCaseToSuiteMessage);

    if (Action == CheckAction::Rename)
      Diag << FixItHint::CreateReplacement(
          CharSourceRange::getTokenRange(Loc, Loc), *Replacement);
  }

  bool ReplacementFound = false;
  UpgradeGoogletestCaseCheck *Check;
  Preprocessor *PP;
};

} // namespace

void UpgradeGoogletestCaseCheck::registerPPCallbacks(const SourceManager &,
                                                     Preprocessor *PP,
                                                     Preprocessor *) {
  PP->addPPCallbacks(
      std::make_unique<UpgradeGoogletestCasePPCallback>(this, PP));
}

void UpgradeGoogletestCaseCheck::registerMatchers(MatchFinder *Finder) {
  auto LocationFilter =
      unless(isExpansionInFileMatching("gtest/gtest(-typed-test)?\\.h$"));

  // Matchers for the member functions that are being renamed. In each matched
  // Google Test class, we check for the existence of one new method name. This
  // makes sure the check gives warnings only if the included version of Google
  // Test is recent enough.
  auto Methods =
      cxxMethodDecl(
          anyOf(
              cxxMethodDecl(
                  hasAnyName("SetUpTestCase", "TearDownTestCase"),
                  ofClass(
                      cxxRecordDecl(isSameOrDerivedFrom(cxxRecordDecl(
                                        hasName("::testing::Test"),
                                        hasMethod(hasName("SetUpTestSuite")))))
                          .bind("class"))),
              cxxMethodDecl(
                  hasName("test_case_name"),
                  ofClass(
                      cxxRecordDecl(isSameOrDerivedFrom(cxxRecordDecl(
                                        hasName("::testing::TestInfo"),
                                        hasMethod(hasName("test_suite_name")))))
                          .bind("class"))),
              cxxMethodDecl(
                  hasAnyName("OnTestCaseStart", "OnTestCaseEnd"),
                  ofClass(cxxRecordDecl(
                              isSameOrDerivedFrom(cxxRecordDecl(
                                  hasName("::testing::TestEventListener"),
                                  hasMethod(hasName("OnTestSuiteStart")))))
                              .bind("class"))),
              cxxMethodDecl(
                  hasAnyName("current_test_case", "successful_test_case_count",
                             "failed_test_case_count", "total_test_case_count",
                             "test_case_to_run_count", "GetTestCase"),
                  ofClass(cxxRecordDecl(
                              isSameOrDerivedFrom(cxxRecordDecl(
                                  hasName("::testing::UnitTest"),
                                  hasMethod(hasName("current_test_suite")))))
                              .bind("class")))))
          .bind("method");

  Finder->addMatcher(expr(anyOf(callExpr(callee(Methods)).bind("call"),
                                declRefExpr(to(Methods)).bind("ref")),
                          LocationFilter),
                     this);

  Finder->addMatcher(
      usingDecl(hasAnyUsingShadowDecl(hasTargetDecl(Methods)), LocationFilter)
          .bind("using"),
      this);

  Finder->addMatcher(cxxMethodDecl(Methods, LocationFilter), this);

  // Matchers for `TestCase` -> `TestSuite`. The fact that `TestCase` is an
  // alias and not a class declaration ensures we only match with a recent
  // enough version of Google Test.
  auto TestCaseTypeAlias =
      typeAliasDecl(hasName("::testing::TestCase")).bind("test-case");
  Finder->addMatcher(
      typeLoc(loc(qualType(typedefType(hasDeclaration(TestCaseTypeAlias)))),
              unless(hasAncestor(decl(isImplicit()))), LocationFilter)
          .bind("typeloc"),
      this);
  Finder->addMatcher(
      usingDecl(hasAnyUsingShadowDecl(hasTargetDecl(TestCaseTypeAlias)))
          .bind("using"),
      this);
  Finder->addMatcher(
      typeLoc(loc(usingType(hasUnderlyingType(
                  typedefType(hasDeclaration(TestCaseTypeAlias))))),
              unless(hasAncestor(decl(isImplicit()))), LocationFilter)
          .bind("typeloc"),
      this);
}

static llvm::StringRef getNewMethodName(llvm::StringRef CurrentName) {
  std::pair<llvm::StringRef, llvm::StringRef> ReplacementMap[] = {
      {"SetUpTestCase", "SetUpTestSuite"},
      {"TearDownTestCase", "TearDownTestSuite"},
      {"test_case_name", "test_suite_name"},
      {"OnTestCaseStart", "OnTestSuiteStart"},
      {"OnTestCaseEnd", "OnTestSuiteEnd"},
      {"current_test_case", "current_test_suite"},
      {"successful_test_case_count", "successful_test_suite_count"},
      {"failed_test_case_count", "failed_test_suite_count"},
      {"total_test_case_count", "total_test_suite_count"},
      {"test_case_to_run_count", "test_suite_to_run_count"},
      {"GetTestCase", "GetTestSuite"}};

  for (auto &Mapping : ReplacementMap) {
    if (CurrentName == Mapping.first)
      return Mapping.second;
  }

  llvm_unreachable("Unexpected function name");
}

template <typename NodeType>
static bool isInInstantiation(const NodeType &Node,
                              const MatchFinder::MatchResult &Result) {
  return !match(isInTemplateInstantiation(), Node, *Result.Context).empty();
}

template <typename NodeType>
static bool isInTemplate(const NodeType &Node,
                         const MatchFinder::MatchResult &Result) {
  internal::Matcher<NodeType> IsInsideTemplate =
      hasAncestor(decl(anyOf(classTemplateDecl(), functionTemplateDecl())));
  return !match(IsInsideTemplate, Node, *Result.Context).empty();
}

static bool
derivedTypeHasReplacementMethod(const MatchFinder::MatchResult &Result,
                                llvm::StringRef ReplacementMethod) {
  const auto *Class = Result.Nodes.getNodeAs<CXXRecordDecl>("class");
  return !match(cxxRecordDecl(
                    unless(isExpansionInFileMatching(
                        "gtest/gtest(-typed-test)?\\.h$")),
                    hasMethod(cxxMethodDecl(hasName(ReplacementMethod)))),
                *Class, *Result.Context)
              .empty();
}

static CharSourceRange
getAliasNameRange(const MatchFinder::MatchResult &Result) {
  if (const auto *Using = Result.Nodes.getNodeAs<UsingDecl>("using")) {
    return CharSourceRange::getTokenRange(
        Using->getNameInfo().getSourceRange());
  }
  TypeLoc TL = *Result.Nodes.getNodeAs<TypeLoc>("typeloc");
  if (auto QTL = TL.getAs<QualifiedTypeLoc>())
    TL = QTL.getUnqualifiedLoc();

  if (auto TTL = TL.getAs<TypedefTypeLoc>())
    return CharSourceRange::getTokenRange(TTL.getNameLoc());
  return CharSourceRange::getTokenRange(TL.castAs<UsingTypeLoc>().getNameLoc());
}

void UpgradeGoogletestCaseCheck::check(const MatchFinder::MatchResult &Result) {
  llvm::StringRef ReplacementText;
  CharSourceRange ReplacementRange;
  if (const auto *Method = Result.Nodes.getNodeAs<CXXMethodDecl>("method")) {
    ReplacementText = getNewMethodName(Method->getName());

    bool IsInInstantiation = false;
    bool IsInTemplate = false;
    bool AddFix = true;
    if (const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call")) {
      const auto *Callee = llvm::cast<MemberExpr>(Call->getCallee());
      ReplacementRange = CharSourceRange::getTokenRange(Callee->getMemberLoc(),
                                                        Callee->getMemberLoc());
      IsInInstantiation = isInInstantiation(*Call, Result);
      IsInTemplate = isInTemplate<Stmt>(*Call, Result);
    } else if (const auto *Ref = Result.Nodes.getNodeAs<DeclRefExpr>("ref")) {
      ReplacementRange =
          CharSourceRange::getTokenRange(Ref->getNameInfo().getSourceRange());
      IsInInstantiation = isInInstantiation(*Ref, Result);
      IsInTemplate = isInTemplate<Stmt>(*Ref, Result);
    } else if (const auto *Using = Result.Nodes.getNodeAs<UsingDecl>("using")) {
      ReplacementRange =
          CharSourceRange::getTokenRange(Using->getNameInfo().getSourceRange());
      IsInInstantiation = isInInstantiation(*Using, Result);
      IsInTemplate = isInTemplate<Decl>(*Using, Result);
    } else {
      // This branch means we have matched a function declaration / definition
      // either for a function from googletest or for a function in a derived
      // class.

      ReplacementRange = CharSourceRange::getTokenRange(
          Method->getNameInfo().getSourceRange());
      IsInInstantiation = isInInstantiation(*Method, Result);
      IsInTemplate = isInTemplate<Decl>(*Method, Result);

      // If the type of the matched method is strictly derived from a googletest
      // type and has both the old and new member function names, then we cannot
      // safely rename (or delete) the old name version.
      AddFix = !derivedTypeHasReplacementMethod(Result, ReplacementText);
    }

    if (IsInInstantiation) {
      if (!MatchedTemplateLocations.contains(ReplacementRange.getBegin())) {
        // For each location matched in a template instantiation, we check if
        // the location can also be found in `MatchedTemplateLocations`. If it
        // is not found, that means the expression did not create a match
        // without the instantiation and depends on template parameters. A
        // manual fix is probably required so we provide only a warning.
        diag(ReplacementRange.getBegin(), RenameCaseToSuiteMessage);
      }
      return;
    }

    if (IsInTemplate) {
      // We gather source locations from template matches not in template
      // instantiations for future matches.
      MatchedTemplateLocations.insert(ReplacementRange.getBegin());
    }

    if (!AddFix) {
      diag(ReplacementRange.getBegin(), RenameCaseToSuiteMessage);
      return;
    }
  } else {
    // This is a match for `TestCase` to `TestSuite` refactoring.
    assert(Result.Nodes.getNodeAs<TypeAliasDecl>("test-case") != nullptr);
    ReplacementText = "TestSuite";
    ReplacementRange = getAliasNameRange(Result);

    // We do not need to keep track of template instantiations for this branch,
    // because we are matching a `TypeLoc` for the alias declaration. Templates
    // will only be instantiated with the true type name, `TestSuite`.
  }

  DiagnosticBuilder Diag =
      diag(ReplacementRange.getBegin(), RenameCaseToSuiteMessage);

  ReplacementRange = Lexer::makeFileCharRange(
      ReplacementRange, *Result.SourceManager, Result.Context->getLangOpts());
  if (ReplacementRange.isInvalid())
    // An invalid source range likely means we are inside a macro body. A manual
    // fix is likely needed so we do not create a fix-it hint.
    return;

  Diag << FixItHint::CreateReplacement(ReplacementRange, ReplacementText);
}

} // namespace clang::tidy::google
