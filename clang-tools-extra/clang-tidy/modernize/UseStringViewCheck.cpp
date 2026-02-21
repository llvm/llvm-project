//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStringViewCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/StringMap.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

static constexpr StringRef StringViewClassKey = "string";
static constexpr StringRef WStringViewClassKey = "wstring";
static constexpr StringRef U8StringViewClassKey = "u8string";
static constexpr StringRef U16StringViewClassKey = "u16string";
static constexpr StringRef U32StringViewClassKey = "u32string";

static auto getStringTypeMatcher(StringRef CharType) {
  return hasCanonicalType(hasDeclaration(cxxRecordDecl(hasName(CharType))));
}

static void fixReturns(const FunctionDecl *FuncDecl, DiagnosticBuilder &Diag,
                       ASTContext &Context) {
  auto Matches = match(
      findAll(returnStmt(hasReturnValue(ignoringParenImpCasts(
          cxxTemporaryObjectExpr(argumentCountIs(0)).bind("temp_obj_expr"))))),
      *FuncDecl->getBody(), Context);

  for (const auto &Match : Matches)
    if (const auto *TempObjExpr =
            Match.getNodeAs<CXXTemporaryObjectExpr>("temp_obj_expr");
        TempObjExpr && TempObjExpr->getSourceRange().isValid())
      Diag << FixItHint::CreateReplacement(TempObjExpr->getSourceRange(), "{}");
}

UseStringViewCheck::UseStringViewCheck(StringRef Name,
                                       ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoredFunctions(utils::options::parseStringList(
          Options.get("IgnoredFunctions", "toString$;ToString$;to_string$"))) {
  parseReplacementStringViewClass(
      Options.get("ReplacementStringViewClass", ""));
}

void UseStringViewCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoredFunctions",
                utils::options::serializeStringList(IgnoredFunctions));
  Options.store(Opts, "ReplacementStringViewClass",
                (Twine("") + StringViewClassKey + "=" + StringViewClass + ";" +
                 WStringViewClassKey + "=" + WStringViewClass + ";" +
                 U8StringViewClassKey + "=" + U8StringViewClass + ";" +
                 U16StringViewClassKey + "=" + U16StringViewClass + ";" +
                 U32StringViewClassKey + "=" + U32StringViewClass)
                    .str());
}

void UseStringViewCheck::registerMatchers(MatchFinder *Finder) {
  const auto IsStdString = getStringTypeMatcher("::std::basic_string");
  // TODO: also consider *StringViewClass types
  const auto IsStdStringView = getStringTypeMatcher("::std::basic_string_view");
  const auto IgnoredFunctionsMatcher =
      matchers::matchesAnyListedRegexName(IgnoredFunctions);
  const auto TernaryOperator = conditionalOperator(
      hasTrueExpression(ignoringParenImpCasts(stringLiteral())),
      hasFalseExpression(ignoringParenImpCasts(stringLiteral())));
  const auto VirtualOrOperator =
      cxxMethodDecl(anyOf(cxxConversionDecl(), isVirtual()));
  Finder->addMatcher(
      functionDecl(
          isDefinition(),
          unless(anyOf(VirtualOrOperator, IgnoredFunctionsMatcher,
                       ast_matchers::isExplicitTemplateSpecialization())),
          returns(IsStdString), hasDescendant(returnStmt()),
          unless(hasDescendant(returnStmt(hasReturnValue(unless(
              anyOf(stringLiteral(), hasType(IsStdStringView), TernaryOperator,
                    cxxConstructExpr(anyOf(
                        allOf(hasType(IsStdString), argumentCountIs(0)),
                        allOf(isListInitialization(),
                              unless(cxxTemporaryObjectExpr()),
                              hasArgument(0, ignoringParenImpCasts(
                                                 stringLiteral()))))))))))))
          .bind("func"),
      this);
}

void UseStringViewCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("func");
  assert(MatchedDecl);
  bool ShouldAKA = false;
  const std::string DesugaredTypeStr =
      clang::desugarForDiagnostic(
          *Result.Context, QualType(MatchedDecl->getReturnType()), ShouldAKA)
          .getAsString();
  const StringRef DestReturnTypeStr = toStringViewTypeStr(DesugaredTypeStr);

  auto Diag =
      diag(MatchedDecl->getTypeSpecStartLoc(),
           "consider using '%0' to avoid unnecessary copying and allocations")
      << DestReturnTypeStr;

  fixReturns(MatchedDecl, Diag, *Result.Context);

  for (const auto *FuncDecl : MatchedDecl->redecls())
    if (const SourceRange ReturnTypeRange =
            FuncDecl->getReturnTypeSourceRange();
        ReturnTypeRange.isValid())
      Diag << FixItHint::CreateReplacement(ReturnTypeRange, DestReturnTypeStr);
}

StringRef UseStringViewCheck::toStringViewTypeStr(StringRef Type) const {
  if (Type.contains("wchar_t"))
    return WStringViewClass;
  if (Type.contains("char8_t"))
    return U8StringViewClass;
  if (Type.contains("char16_t"))
    return U16StringViewClass;
  if (Type.contains("char32_t"))
    return U32StringViewClass;
  return StringViewClass;
}

void UseStringViewCheck::parseReplacementStringViewClass(StringRef Options) {
  if (Options.empty())
    return;
  const llvm::StringMap<StringRef *> StringClassesMap{
      {StringViewClassKey, &StringViewClass},
      {WStringViewClassKey, &WStringViewClass},
      {U8StringViewClassKey, &U8StringViewClass},
      {U16StringViewClassKey, &U16StringViewClass},
      {U32StringViewClassKey, &U32StringViewClass}};
  for (const auto &Option : utils::options::parseStringList(Options)) {
    const auto Split = Option.split('=');
    if (auto It = StringClassesMap.find(Split.first);
        It != StringClassesMap.end())
      *It->second = Split.second;
  }
}

} // namespace clang::tidy::modernize
