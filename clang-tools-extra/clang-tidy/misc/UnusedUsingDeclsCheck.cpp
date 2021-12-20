//===--- UnusedUsingDeclsCheck.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnusedUsingDeclsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

namespace {
// FIXME: Move ASTMatcher library.
AST_POLYMORPHIC_MATCHER_P(
    forEachTemplateArgument,
    AST_POLYMORPHIC_SUPPORTED_TYPES(ClassTemplateSpecializationDecl,
                                    TemplateSpecializationType, FunctionDecl),
    clang::ast_matchers::internal::Matcher<TemplateArgument>, InnerMatcher) {
  ArrayRef<TemplateArgument> TemplateArgs =
      clang::ast_matchers::internal::getTemplateSpecializationArgs(Node);
  clang::ast_matchers::internal::BoundNodesTreeBuilder Result;
  bool Matched = false;
  for (const auto &Arg : TemplateArgs) {
    clang::ast_matchers::internal::BoundNodesTreeBuilder ArgBuilder(*Builder);
    if (InnerMatcher.matches(Arg, Finder, &ArgBuilder)) {
      Matched = true;
      Result.addMatch(ArgBuilder);
    }
  }
  *Builder = std::move(Result);
  return Matched;
}

AST_MATCHER_P(DeducedTemplateSpecializationType, refsToTemplatedDecl,
              clang::ast_matchers::internal::Matcher<NamedDecl>, DeclMatcher) {
  if (const auto *TD = Node.getTemplateName().getAsTemplateDecl())
    return DeclMatcher.matches(*TD, Finder, Builder);
  return false;
}
} // namespace

// A function that helps to tell whether a TargetDecl in a UsingDecl will be
// checked. Only variable, function, function template, class template, class,
// enum declaration and enum constant declaration are considered.
static bool shouldCheckDecl(const Decl *TargetDecl) {
  return isa<RecordDecl>(TargetDecl) || isa<ClassTemplateDecl>(TargetDecl) ||
         isa<FunctionDecl>(TargetDecl) || isa<VarDecl>(TargetDecl) ||
         isa<FunctionTemplateDecl>(TargetDecl) || isa<EnumDecl>(TargetDecl) ||
         isa<EnumConstantDecl>(TargetDecl);
}

void UnusedUsingDeclsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(usingDecl(isExpansionInMainFile()).bind("using"), this);
  auto DeclMatcher = hasDeclaration(namedDecl().bind("used"));
  Finder->addMatcher(loc(enumType(DeclMatcher)), this);
  Finder->addMatcher(loc(recordType(DeclMatcher)), this);
  Finder->addMatcher(loc(templateSpecializationType(DeclMatcher)), this);
  Finder->addMatcher(loc(deducedTemplateSpecializationType(
                         refsToTemplatedDecl(namedDecl().bind("used")))),
                     this);
  Finder->addMatcher(declRefExpr().bind("used"), this);
  Finder->addMatcher(callExpr(callee(unresolvedLookupExpr().bind("used"))),
                     this);
  Finder->addMatcher(
      callExpr(hasDeclaration(functionDecl(
          forEachTemplateArgument(templateArgument().bind("used"))))),
      this);
  Finder->addMatcher(loc(templateSpecializationType(forEachTemplateArgument(
                         templateArgument().bind("used")))),
                     this);
}

void UnusedUsingDeclsCheck::check(const MatchFinder::MatchResult &Result) {
  if (Result.Context->getDiagnostics().hasUncompilableErrorOccurred())
    return;

  if (const auto *Using = Result.Nodes.getNodeAs<UsingDecl>("using")) {
    // Ignores using-declarations defined in macros.
    if (Using->getLocation().isMacroID())
      return;

    // Ignores using-declarations defined in class definition.
    if (isa<CXXRecordDecl>(Using->getDeclContext()))
      return;

    // FIXME: We ignore using-decls defined in function definitions at the
    // moment because of false positives caused by ADL and different function
    // scopes.
    if (isa<FunctionDecl>(Using->getDeclContext()))
      return;

    UsingDeclContext Context(Using);
    Context.UsingDeclRange = CharSourceRange::getCharRange(
        Using->getBeginLoc(),
        Lexer::findLocationAfterToken(
            Using->getEndLoc(), tok::semi, *Result.SourceManager, getLangOpts(),
            /*SkipTrailingWhitespaceAndNewLine=*/true));
    for (const auto *UsingShadow : Using->shadows()) {
      const auto *TargetDecl = UsingShadow->getTargetDecl()->getCanonicalDecl();
      if (shouldCheckDecl(TargetDecl))
        Context.UsingTargetDecls.insert(TargetDecl);
    }
    if (!Context.UsingTargetDecls.empty())
      Contexts.push_back(Context);
    return;
  }

  // Mark a corresponding using declaration as used.
  auto RemoveNamedDecl = [&](const NamedDecl *Used) {
    removeFromFoundDecls(Used);
    // Also remove variants of Used.
    if (const auto *FD = dyn_cast<FunctionDecl>(Used)) {
      removeFromFoundDecls(FD->getPrimaryTemplate());
    } else if (const auto *Specialization =
                   dyn_cast<ClassTemplateSpecializationDecl>(Used)) {
      removeFromFoundDecls(Specialization->getSpecializedTemplate());
    } else if (const auto *FD = dyn_cast<FunctionDecl>(Used)) {
      if (const auto *FDT = FD->getPrimaryTemplate())
        removeFromFoundDecls(FDT);
    } else if (const auto *ECD = dyn_cast<EnumConstantDecl>(Used)) {
      if (const auto *ET = ECD->getType()->getAs<EnumType>())
        removeFromFoundDecls(ET->getDecl());
    }
  };
  // We rely on the fact that the clang AST is walked in order, usages are only
  // marked after a corresponding using decl has been found.
  if (const auto *Used = Result.Nodes.getNodeAs<NamedDecl>("used")) {
    RemoveNamedDecl(Used);
    return;
  }

  if (const auto *Used = Result.Nodes.getNodeAs<TemplateArgument>("used")) {
    if (Used->getKind() == TemplateArgument::Template) {
      if (const auto *TD = Used->getAsTemplate().getAsTemplateDecl())
        removeFromFoundDecls(TD);
    } else if (Used->getKind() == TemplateArgument::Type) {
      if (auto *RD = Used->getAsType()->getAsCXXRecordDecl())
        removeFromFoundDecls(RD);
    } else if (Used->getKind() == TemplateArgument::Declaration) {
      RemoveNamedDecl(Used->getAsDecl());
    }
    return;
  }

  if (const auto *DRE = Result.Nodes.getNodeAs<DeclRefExpr>("used")) {
    RemoveNamedDecl(DRE->getDecl());
    return;
  }
  // Check the uninstantiated template function usage.
  if (const auto *ULE = Result.Nodes.getNodeAs<UnresolvedLookupExpr>("used")) {
    for (const NamedDecl *ND : ULE->decls()) {
      if (const auto *USD = dyn_cast<UsingShadowDecl>(ND))
        removeFromFoundDecls(USD->getTargetDecl()->getCanonicalDecl());
    }
  }
}

void UnusedUsingDeclsCheck::removeFromFoundDecls(const Decl *D) {
  if (!D)
    return;
  // FIXME: Currently, we don't handle the using-decls being used in different
  // scopes (such as different namespaces, different functions). Instead of
  // giving an incorrect message, we mark all of them as used.
  //
  // FIXME: Use a more efficient way to find a matching context.
  for (auto &Context : Contexts) {
    if (Context.UsingTargetDecls.contains(D->getCanonicalDecl()))
      Context.IsUsed = true;
  }
}

void UnusedUsingDeclsCheck::onEndOfTranslationUnit() {
  for (const auto &Context : Contexts) {
    if (!Context.IsUsed) {
      diag(Context.FoundUsingDecl->getLocation(), "using decl %0 is unused")
          << Context.FoundUsingDecl;
      // Emit a fix and a fix description of the check;
      diag(Context.FoundUsingDecl->getLocation(),
           /*Description=*/"remove the using", DiagnosticIDs::Note)
          << FixItHint::CreateRemoval(Context.UsingDeclRange);
    }
  }
  Contexts.clear();
}

} // namespace misc
} // namespace tidy
} // namespace clang
