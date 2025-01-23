#include "MakeFunctionToDirectCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void MakeFunctionToDirectCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus17)
    return;
  // Match make_xxx function calls
  Finder->addMatcher(callExpr(callee(functionDecl(hasAnyName(
                                  "std::make_optional", "std::make_unique",
                                  "std::make_shared", "std::make_pair"))))
                         .bind("make_call"),
                     this);
}

bool MakeFunctionToDirectCheck::isMakeFunction(
    const std::string &FuncName) const {
  static const std::array<std::string_view, 4> MakeFuncs = {
      "make_optional", "make_unique", "make_shared", "make_pair"};

  return std::any_of(MakeFuncs.begin(), MakeFuncs.end(),
                     [&](const auto &Prefix) {
                       return FuncName.find(Prefix) != std::string::npos;
                     });
}

std::string MakeFunctionToDirectCheck::getTemplateType(
    const CXXConstructExpr *Construct) const {
  if (!Construct)
    return {};

  const auto *RecordType =
      dyn_cast<clang::RecordType>(Construct->getType().getTypePtr());
  if (!RecordType)
    return {};

  return RecordType->getDecl()->getNameAsString();
}

void MakeFunctionToDirectCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("make_call");
  if (!Call)
    return;

  const auto *FuncDecl = dyn_cast<FunctionDecl>(Call->getCalleeDecl());
  if (!FuncDecl || !FuncDecl->getTemplateSpecializationArgs())
    return;

  std::string FuncName = FuncDecl->getNameAsString();
  if (!isMakeFunction(FuncName))
    return;

  std::string Args;
  if (Call->getNumArgs() > 0) {
    SourceRange ArgRange(Call->getArg(0)->getBeginLoc(),
                         Call->getArg(Call->getNumArgs() - 1)->getEndLoc());
    Args = std::string(Lexer::getSourceText(
        CharSourceRange::getTokenRange(ArgRange), *Result.SourceManager,
        Result.Context->getLangOpts()));
  }

  std::string Replacement;
  if (FuncName == "make_unique" || FuncName == "make_shared") {
    const auto *TemplateArgs = FuncDecl->getTemplateSpecializationArgs();
    if (!TemplateArgs || TemplateArgs->size() == 0)
      return;

    QualType Type = TemplateArgs->get(0).getAsType();
    PrintingPolicy Policy(Result.Context->getLangOpts());
    Policy.SuppressTagKeyword = true;
    std::string TypeStr = Type.getAsString(Policy);

    std::string SmartPtr =
        (FuncName == "make_unique") ? "unique_ptr" : "shared_ptr";
    Replacement = "std::" + SmartPtr + "(new " + TypeStr + "(" + Args + "))";
  } else {
    std::string TemplateType;
    if (FuncName == "make_optional")
      TemplateType = "std::optional";
    else if (FuncName == "make_shared")
      TemplateType = "std::shared_ptr";
    else if (FuncName == "make_pair")
      TemplateType = "std::pair";

    if (TemplateType.empty())
      return;

    Replacement = TemplateType + "(" + Args + ")";
  }

  if (!Replacement.empty()) {
    diag(Call->getBeginLoc(),
         "use class template argument deduction (CTAD) instead of %0")
        << FuncName
        << FixItHint::CreateReplacement(
               CharSourceRange::getTokenRange(Call->getSourceRange()),
               Replacement);
  }
}

} // namespace clang::tidy::modernize