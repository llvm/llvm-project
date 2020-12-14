//===--- ReplaceAutoPtrCheck.cpp - clang-tidy------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReplaceAutoPtrCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

namespace {
static const char AutoPtrTokenId[] = "AutoPrTokenId";
static const char AutoPtrOwnershipTransferId[] = "AutoPtrOwnershipTransferId";

/// Matches expressions that are lvalues.
///
/// In the following example, a[0] matches expr(isLValue()):
/// \code
///   std::string a[2];
///   std::string b;
///   b = a[0];
///   b = "this string won't match";
/// \endcode
AST_MATCHER(Expr, isLValue) { return Node.getValueKind() == VK_LValue; }

} // namespace

ReplaceAutoPtrCheck::ReplaceAutoPtrCheck(StringRef Name,
                                         ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM)) {}

void ReplaceAutoPtrCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
}

void ReplaceAutoPtrCheck::registerMatchers(MatchFinder *Finder) {
  auto AutoPtrDecl = recordDecl(hasName("auto_ptr"), isInStdNamespace());
  auto AutoPtrType = qualType(hasDeclaration(AutoPtrDecl));

  //   std::auto_ptr<int> a;
  //        ^~~~~~~~~~~~~
  //
  //   typedef std::auto_ptr<int> int_ptr_t;
  //                ^~~~~~~~~~~~~
  //
  //   std::auto_ptr<int> fn(std::auto_ptr<int>);
  //        ^~~~~~~~~~~~~         ^~~~~~~~~~~~~
  Finder->addMatcher(typeLoc(loc(qualType(AutoPtrType,
                                          // Skip elaboratedType() as the named
                                          // type will match soon thereafter.
                                          unless(elaboratedType()))))
                         .bind(AutoPtrTokenId),
                     this);

  //   using std::auto_ptr;
  //   ^~~~~~~~~~~~~~~~~~~
  Finder->addMatcher(usingDecl(hasAnyUsingShadowDecl(hasTargetDecl(namedDecl(
                                   hasName("auto_ptr"), isInStdNamespace()))))
                         .bind(AutoPtrTokenId),
                     this);

  // Find ownership transfers via copy construction and assignment.
  // AutoPtrOwnershipTransferId is bound to the part that has to be wrapped
  // into std::move().
  //   std::auto_ptr<int> i, j;
  //   i = j;
  //   ~~~~^
  auto MovableArgumentMatcher =
      expr(isLValue(), hasType(AutoPtrType)).bind(AutoPtrOwnershipTransferId);

  Finder->addMatcher(
      cxxOperatorCallExpr(hasOverloadedOperatorName("="),
                          callee(cxxMethodDecl(ofClass(AutoPtrDecl))),
                          hasArgument(1, MovableArgumentMatcher)),
      this);
  Finder->addMatcher(
      traverse(TK_AsIs,
               cxxConstructExpr(hasType(AutoPtrType), argumentCountIs(1),
                                hasArgument(0, MovableArgumentMatcher))),
      this);
}

void ReplaceAutoPtrCheck::registerPPCallbacks(const SourceManager &SM,
                                              Preprocessor *PP,
                                              Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void ReplaceAutoPtrCheck::check(const MatchFinder::MatchResult &Result) {
  SourceManager &SM = *Result.SourceManager;
  if (const auto *E =
          Result.Nodes.getNodeAs<Expr>(AutoPtrOwnershipTransferId)) {
    CharSourceRange Range = Lexer::makeFileCharRange(
        CharSourceRange::getTokenRange(E->getSourceRange()), SM, LangOptions());

    if (Range.isInvalid())
      return;

    auto Diag = diag(Range.getBegin(), "use std::move to transfer ownership")
                << FixItHint::CreateInsertion(Range.getBegin(), "std::move(")
                << FixItHint::CreateInsertion(Range.getEnd(), ")")
                << Inserter.createMainFileIncludeInsertion("<utility>");

    return;
  }

  SourceLocation AutoPtrLoc;
  if (const auto *TL = Result.Nodes.getNodeAs<TypeLoc>(AutoPtrTokenId)) {
    //   std::auto_ptr<int> i;
    //        ^
    if (auto Loc = TL->getAs<TemplateSpecializationTypeLoc>())
      AutoPtrLoc = Loc.getTemplateNameLoc();
  } else if (const auto *D =
                 Result.Nodes.getNodeAs<UsingDecl>(AutoPtrTokenId)) {
    // using std::auto_ptr;
    //            ^
    AutoPtrLoc = D->getNameInfo().getBeginLoc();
  } else {
    llvm_unreachable("Bad Callback. No node provided.");
  }

  if (AutoPtrLoc.isMacroID())
    AutoPtrLoc = SM.getSpellingLoc(AutoPtrLoc);

  // Ensure that only the 'auto_ptr' token is replaced and not the template
  // aliases.
  if (StringRef(SM.getCharacterData(AutoPtrLoc), strlen("auto_ptr")) !=
      "auto_ptr")
    return;

  SourceLocation EndLoc =
      AutoPtrLoc.getLocWithOffset(strlen("auto_ptr") - 1);
  diag(AutoPtrLoc, "auto_ptr is deprecated, use unique_ptr instead")
      << FixItHint::CreateReplacement(SourceRange(AutoPtrLoc, EndLoc),
                                      "unique_ptr");
}

} // namespace modernize
} // namespace tidy
} // namespace clang
