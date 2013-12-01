//===-- PassByValueActions.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the definition of the ASTMatcher callback for the
/// PassByValue transform.
///
//===----------------------------------------------------------------------===//

#include "PassByValueActions.h"
#include "PassByValueMatchers.h"
#include "Core/IncludeDirectives.h"
#include "Core/Transform.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

namespace {
/// \brief \c clang::RecursiveASTVisitor that checks that the given
/// \c ParmVarDecl is used exactly one time.
///
/// \see ExactlyOneUsageVisitor::hasExactlyOneUsageIn()
class ExactlyOneUsageVisitor
    : public RecursiveASTVisitor<ExactlyOneUsageVisitor> {
  friend class RecursiveASTVisitor<ExactlyOneUsageVisitor>;

public:
  ExactlyOneUsageVisitor(const ParmVarDecl *ParamDecl) : ParamDecl(ParamDecl) {}

  /// \brief Whether or not the parameter variable is referred only once in the
  /// given constructor.
  bool hasExactlyOneUsageIn(const CXXConstructorDecl *Ctor) {
    Count = 0;
    TraverseDecl(const_cast<CXXConstructorDecl *>(Ctor));
    return Count == 1;
  }

private:
  /// \brief Counts the number of references to a variable.
  ///
  /// Stops the AST traversal if more than one usage is found.
  bool VisitDeclRefExpr(DeclRefExpr *D) {
    if (const ParmVarDecl *To = llvm::dyn_cast<ParmVarDecl>(D->getDecl()))
      if (To == ParamDecl) {
        ++Count;
        if (Count > 1)
          // no need to look further, used more than once
          return false;
      }
    return true;
  }

  const ParmVarDecl *ParamDecl;
  unsigned Count;
};
} // end anonymous namespace

/// \brief Whether or not \p ParamDecl is used exactly one time in \p Ctor.
///
/// Checks both in the init-list and the body of the constructor.
static bool paramReferredExactlyOnce(const CXXConstructorDecl *Ctor,
                                     const ParmVarDecl *ParamDecl) {
  ExactlyOneUsageVisitor Visitor(ParamDecl);
  return Visitor.hasExactlyOneUsageIn(Ctor);
}

/// \brief Find all references to \p ParamDecl across all of the
/// redeclarations of \p Ctor.
static void
collectParamDecls(const CXXConstructorDecl *Ctor, const ParmVarDecl *ParamDecl,
                  llvm::SmallVectorImpl<const ParmVarDecl *> &Results) {
  unsigned ParamIdx = ParamDecl->getFunctionScopeIndex();

  for (CXXConstructorDecl::redecl_iterator I = Ctor->redecls_begin(),
                                           E = Ctor->redecls_end();
       I != E; ++I)
    Results.push_back((*I)->getParamDecl(ParamIdx));
}

void ConstructorParamReplacer::run(const MatchFinder::MatchResult &Result) {
  assert(IncludeManager && "Include directives manager not set.");
  SourceManager &SM = *Result.SourceManager;
  const CXXConstructorDecl *Ctor =
      Result.Nodes.getNodeAs<CXXConstructorDecl>(PassByValueCtorId);
  const ParmVarDecl *ParamDecl =
      Result.Nodes.getNodeAs<ParmVarDecl>(PassByValueParamId);
  const CXXCtorInitializer *Initializer =
      Result.Nodes.getNodeAs<CXXCtorInitializer>(PassByValueInitializerId);
  assert(Ctor && ParamDecl && Initializer && "Bad Callback, missing node.");

  // Check this now to avoid unnecessary work. The param locations are checked
  // later.
  if (!Owner.isFileModifiable(SM, Initializer->getSourceLocation()))
    return;

  // The parameter will be in an unspecified state after the move, so check if
  // the parameter is used for anything else other than the copy. If so do not
  // apply any changes.
  if (!paramReferredExactlyOnce(Ctor, ParamDecl))
    return;

  llvm::SmallVector<const ParmVarDecl *, 2> AllParamDecls;
  collectParamDecls(Ctor, ParamDecl, AllParamDecls);

  // Generate all replacements for the params.
  llvm::SmallVector<Replacement, 2> ParamReplaces;
  for (unsigned I = 0, E = AllParamDecls.size(); I != E; ++I) {
    TypeLoc ParamTL = AllParamDecls[I]->getTypeSourceInfo()->getTypeLoc();
    ReferenceTypeLoc RefTL = ParamTL.getAs<ReferenceTypeLoc>();
    SourceRange Range(AllParamDecls[I]->getLocStart(), ParamTL.getLocEnd());
    CharSourceRange CharRange = Lexer::makeFileCharRange(
        CharSourceRange::getTokenRange(Range), SM, LangOptions());

    // do not generate a replacement when the parameter is already a value
    if (RefTL.isNull())
      continue;

    // transform non-value parameters (e.g: const-ref) to values
    TypeLoc ValueTypeLoc = RefTL.getPointeeLoc();
    llvm::SmallString<32> ValueStr = Lexer::getSourceText(
        CharSourceRange::getTokenRange(ValueTypeLoc.getSourceRange()), SM,
        LangOptions());

    // If it's impossible to change one of the parameter (e.g: comes from an
    // unmodifiable header) quit the callback now, do not generate any changes.
    if (CharRange.isInvalid() || ValueStr.empty() ||
        !Owner.isFileModifiable(SM, CharRange.getBegin()))
      return;

    // 'const Foo &param' -> 'Foo param'
    //  ~~~~~~~~~~~           ~~~^
    ValueStr += ' ';
    ParamReplaces.push_back(Replacement(SM, CharRange, ValueStr));
  }

  // Reject the changes if the the risk level is not acceptable.
  if (!Owner.isAcceptableRiskLevel(RL_Reasonable)) {
    RejectedChanges++;
    return;
  }

  // if needed, include <utility> in the file that uses std::move()
  const FileEntry *STDMoveFile =
      SM.getFileEntryForID(SM.getFileID(Initializer->getLParenLoc()));
  const tooling::Replacement &IncludeReplace =
      IncludeManager->addAngledInclude(STDMoveFile, "utility");
  if (IncludeReplace.isApplicable()) {
    Owner.addReplacementForCurrentTU(IncludeReplace);
    AcceptedChanges++;
  }

  // const-ref params becomes values (const Foo & -> Foo)
  for (const Replacement *I = ParamReplaces.begin(), *E = ParamReplaces.end();
       I != E; ++I) {
    Owner.addReplacementForCurrentTU(*I);
  }
  AcceptedChanges += ParamReplaces.size();

  // move the value in the init-list
  Owner.addReplacementForCurrentTU(Replacement(
      SM, Initializer->getLParenLoc().getLocWithOffset(1), 0, "std::move("));
  Owner.addReplacementForCurrentTU(
      Replacement(SM, Initializer->getRParenLoc(), 0, ")"));
  AcceptedChanges += 2;
}
