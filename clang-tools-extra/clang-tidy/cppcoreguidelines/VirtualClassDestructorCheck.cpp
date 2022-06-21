//===--- VirtualClassDestructorCheck.cpp - clang-tidy -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VirtualClassDestructorCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include <string>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

AST_MATCHER(CXXRecordDecl, hasPublicVirtualOrProtectedNonVirtualDestructor) {
  // We need to call Node.getDestructor() instead of matching a
  // CXXDestructorDecl. Otherwise, tests will fail for class templates, since
  // the primary template (not the specialization) always gets a non-virtual
  // CXXDestructorDecl in the AST. https://bugs.llvm.org/show_bug.cgi?id=51912
  const CXXDestructorDecl *Destructor = Node.getDestructor();
  if (!Destructor)
    return false;

  return (((Destructor->getAccess() == AccessSpecifier::AS_public) &&
           Destructor->isVirtual()) ||
          ((Destructor->getAccess() == AccessSpecifier::AS_protected) &&
           !Destructor->isVirtual()));
}

void VirtualClassDestructorCheck::registerMatchers(MatchFinder *Finder) {
  ast_matchers::internal::Matcher<CXXRecordDecl> InheritsVirtualMethod =
      hasAnyBase(hasType(cxxRecordDecl(has(cxxMethodDecl(isVirtual())))));

  Finder->addMatcher(
      cxxRecordDecl(
          anyOf(has(cxxMethodDecl(isVirtual())), InheritsVirtualMethod),
          unless(hasPublicVirtualOrProtectedNonVirtualDestructor()))
          .bind("ProblematicClassOrStruct"),
      this);
}

static Optional<CharSourceRange>
getVirtualKeywordRange(const CXXDestructorDecl &Destructor,
                       const SourceManager &SM, const LangOptions &LangOpts) {
  if (Destructor.getLocation().isMacroID())
    return None;

  SourceLocation VirtualBeginLoc = Destructor.getBeginLoc();
  SourceLocation VirtualEndLoc = VirtualBeginLoc.getLocWithOffset(
      Lexer::MeasureTokenLength(VirtualBeginLoc, SM, LangOpts));

  /// Range ends with \c StartOfNextToken so that any whitespace after \c
  /// virtual is included.
  SourceLocation StartOfNextToken =
      Lexer::findNextToken(VirtualEndLoc, SM, LangOpts)->getLocation();

  return CharSourceRange::getCharRange(VirtualBeginLoc, StartOfNextToken);
}

static const AccessSpecDecl *
getPublicASDecl(const CXXRecordDecl &StructOrClass) {
  for (DeclContext::specific_decl_iterator<AccessSpecDecl>
           AS{StructOrClass.decls_begin()},
       ASEnd{StructOrClass.decls_end()};
       AS != ASEnd; ++AS) {
    AccessSpecDecl *ASDecl = *AS;
    if (ASDecl->getAccess() == AccessSpecifier::AS_public)
      return ASDecl;
  }

  return nullptr;
}

static FixItHint
generateUserDeclaredDestructor(const CXXRecordDecl &StructOrClass,
                               const SourceManager &SourceManager) {
  std::string DestructorString;
  SourceLocation Loc;
  bool AppendLineBreak = false;

  const AccessSpecDecl *AccessSpecDecl = getPublicASDecl(StructOrClass);

  if (!AccessSpecDecl) {
    if (StructOrClass.isClass()) {
      Loc = StructOrClass.getEndLoc();
      DestructorString = "public:";
      AppendLineBreak = true;
    } else {
      Loc = StructOrClass.getBraceRange().getBegin().getLocWithOffset(1);
    }
  } else {
    Loc = AccessSpecDecl->getEndLoc().getLocWithOffset(1);
  }

  DestructorString = (llvm::Twine(DestructorString) + "\nvirtual ~" +
                      StructOrClass.getName().str() + "() = default;" +
                      (AppendLineBreak ? "\n" : ""))
                         .str();

  return FixItHint::CreateInsertion(Loc, DestructorString);
}

static std::string getSourceText(const CXXDestructorDecl &Destructor) {
  std::string SourceText;
  llvm::raw_string_ostream DestructorStream(SourceText);
  Destructor.print(DestructorStream);
  return SourceText;
}

static std::string eraseKeyword(std::string &DestructorString,
                                const std::string &Keyword) {
  size_t KeywordIndex = DestructorString.find(Keyword);
  if (KeywordIndex != std::string::npos)
    DestructorString.erase(KeywordIndex, Keyword.length());
  return DestructorString;
}

static FixItHint changePrivateDestructorVisibilityTo(
    const std::string &Visibility, const CXXDestructorDecl &Destructor,
    const SourceManager &SM, const LangOptions &LangOpts) {
  std::string DestructorString =
      (llvm::Twine() + Visibility + ":\n" +
       (Visibility == "public" && !Destructor.isVirtual() ? "virtual " : ""))
          .str();

  std::string OriginalDestructor = getSourceText(Destructor);
  if (Visibility == "protected" && Destructor.isVirtualAsWritten())
    OriginalDestructor = eraseKeyword(OriginalDestructor, "virtual ");

  DestructorString =
      (llvm::Twine(DestructorString) + OriginalDestructor +
       (Destructor.isExplicitlyDefaulted() ? ";\n" : "") + "private:")
          .str();

  /// Semicolons ending an explicitly defaulted destructor have to be deleted.
  /// Otherwise, the left-over semicolon trails the \c private: access
  /// specifier.
  SourceLocation EndLocation;
  if (Destructor.isExplicitlyDefaulted())
    EndLocation =
        utils::lexer::findNextTerminator(Destructor.getEndLoc(), SM, LangOpts)
            .getLocWithOffset(1);
  else
    EndLocation = Destructor.getEndLoc().getLocWithOffset(1);

  auto OriginalDestructorRange =
      CharSourceRange::getCharRange(Destructor.getBeginLoc(), EndLocation);
  return FixItHint::CreateReplacement(OriginalDestructorRange,
                                      DestructorString);
}

void VirtualClassDestructorCheck::check(
    const MatchFinder::MatchResult &Result) {

  const auto *MatchedClassOrStruct =
      Result.Nodes.getNodeAs<CXXRecordDecl>("ProblematicClassOrStruct");

  const CXXDestructorDecl *Destructor = MatchedClassOrStruct->getDestructor();
  if (!Destructor)
    return;

  if (Destructor->getAccess() == AccessSpecifier::AS_private) {
    diag(MatchedClassOrStruct->getLocation(),
         "destructor of %0 is private and prevents using the type")
        << MatchedClassOrStruct;
    diag(MatchedClassOrStruct->getLocation(),
         /*Description=*/"make it public and virtual", DiagnosticIDs::Note)
        << changePrivateDestructorVisibilityTo(
               "public", *Destructor, *Result.SourceManager, getLangOpts());
    diag(MatchedClassOrStruct->getLocation(),
         /*Description=*/"make it protected", DiagnosticIDs::Note)
        << changePrivateDestructorVisibilityTo(
               "protected", *Destructor, *Result.SourceManager, getLangOpts());

    return;
  }

  // Implicit destructors are public and non-virtual for classes and structs.
  bool ProtectedAndVirtual = false;
  FixItHint Fix;

  if (MatchedClassOrStruct->hasUserDeclaredDestructor()) {
    if (Destructor->getAccess() == AccessSpecifier::AS_public) {
      Fix = FixItHint::CreateInsertion(Destructor->getLocation(), "virtual ");
    } else if (Destructor->getAccess() == AccessSpecifier::AS_protected) {
      ProtectedAndVirtual = true;
      if (const auto MaybeRange =
              getVirtualKeywordRange(*Destructor, *Result.SourceManager,
                                     Result.Context->getLangOpts()))
        Fix = FixItHint::CreateRemoval(*MaybeRange);
    }
  } else {
    Fix = generateUserDeclaredDestructor(*MatchedClassOrStruct,
                                         *Result.SourceManager);
  }

  diag(MatchedClassOrStruct->getLocation(),
       "destructor of %0 is %select{public and non-virtual|protected and "
       "virtual}1")
      << MatchedClassOrStruct << ProtectedAndVirtual;
  diag(MatchedClassOrStruct->getLocation(),
       "make it %select{public and virtual|protected and non-virtual}0",
       DiagnosticIDs::Note)
      << ProtectedAndVirtual << Fix;
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
