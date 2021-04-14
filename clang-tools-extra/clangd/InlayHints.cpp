//===--- InlayHints.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "InlayHints.h"
#include "ParsedAST.h"
#include "support/Logger.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"

namespace clang {
namespace clangd {

class InlayHintVisitor : public RecursiveASTVisitor<InlayHintVisitor> {
public:
  InlayHintVisitor(std::vector<InlayHint> &Results, ParsedAST &AST)
      : Results(Results), AST(AST.getASTContext()) {}

  bool VisitCXXConstructExpr(CXXConstructExpr *E) {
    // Weed out constructor calls that don't look like a function call with
    // an argument list, by checking the validity of getParenOrBraceRange().
    // Also weed out std::initializer_list constructors as there are no names
    // for the individual arguments.
    if (!E->getParenOrBraceRange().isValid() ||
        E->isStdInitListInitialization()) {
      return true;
    }

    processCall(E->getParenOrBraceRange().getBegin(), E->getConstructor(),
                {E->getArgs(), E->getNumArgs()});
    return true;
  }

  bool VisitCallExpr(CallExpr *E) {
    // Do not show parameter hints for operator calls written using operator
    // syntax or user-defined literals. (Among other reasons, the resulting
    // hints can look awkard, e.g. the expression can itself be a function
    // argument and then we'd get two hints side by side).
    if (isa<CXXOperatorCallExpr>(E) || isa<UserDefinedLiteral>(E))
      return true;

    processCall(E->getRParenLoc(),
                dyn_cast_or_null<FunctionDecl>(E->getCalleeDecl()),
                {E->getArgs(), E->getNumArgs()});
    return true;
  }

  // FIXME: Handle RecoveryExpr to try to hint some invalid calls.

private:
  // The purpose of Anchor is to deal with macros. It should be the call's
  // opening or closing parenthesis or brace. (Always using the opening would
  // make more sense but CallExpr only exposes the closing.) We heuristically
  // assume that if this location does not come from a macro definition, then
  // the entire argument list likely appears in the main file and can be hinted.
  void processCall(SourceLocation Anchor, const FunctionDecl *Callee,
                   llvm::ArrayRef<const Expr *const> Args) {
    if (Args.size() == 0 || !Callee)
      return;

    // If the anchor location comes from a macro defintion, there's nowhere to
    // put hints.
    if (!AST.getSourceManager().getTopMacroCallerLoc(Anchor).isFileID())
      return;

    // The parameter name of a move or copy constructor is not very interesting.
    if (auto *Ctor = dyn_cast<CXXConstructorDecl>(Callee))
      if (Ctor->isCopyOrMoveConstructor())
        return;

    // FIXME: Exclude setters (i.e. functions with one argument whose name
    // begins with "set"), as their parameter name is also not likely to be
    // interesting.

    // Don't show hints for variadic parameters.
    size_t FixedParamCount = getFixedParamCount(Callee);
    size_t ArgCount = std::min(FixedParamCount, Args.size());

    NameVec ParameterNames = chooseParameterNames(Callee, ArgCount);

    for (size_t I = 0; I < ArgCount; ++I) {
      StringRef Name = ParameterNames[I];
      if (!shouldHint(Args[I], Name))
        continue;

      addInlayHint(Args[I]->getSourceRange(), InlayHintKind::ParameterHint,
                   Name.str() + ": ");
    }
  }

  bool shouldHint(const Expr *Arg, StringRef ParamName) {
    if (ParamName.empty())
      return false;

    // If the argument expression is a single name and it matches the
    // parameter name exactly, omit the hint.
    if (ParamName == getSpelledIdentifier(Arg))
      return false;

    // FIXME: Exclude argument expressions preceded by a /*paramName=*/ comment.

    return true;
  }

  // If "E" spells a single unqualified identifier, return that name.
  // Otherwise, return an empty string.
  static StringRef getSpelledIdentifier(const Expr *E) {
    E = E->IgnoreUnlessSpelledInSource();

    if (auto *DRE = dyn_cast<DeclRefExpr>(E))
      if (!DRE->getQualifier())
        return getSimpleName(*DRE->getDecl());

    if (auto *ME = dyn_cast<MemberExpr>(E))
      if (!ME->getQualifier() && ME->isImplicitAccess())
        return getSimpleName(*ME->getMemberDecl());

    return {};
  }

  using NameVec = SmallVector<StringRef, 8>;

  NameVec chooseParameterNames(const FunctionDecl *Callee, size_t ArgCount) {
    // The current strategy here is to use all the parameter names from the
    // canonical declaration, unless they're all empty, in which case we
    // use all the parameter names from the definition (in present in the
    // translation unit).
    // We could try a bit harder, e.g.:
    //   - try all re-declarations, not just canonical + definition
    //   - fall back arg-by-arg rather than wholesale

    NameVec ParameterNames = getParameterNamesForDecl(Callee, ArgCount);

    if (llvm::all_of(ParameterNames, std::mem_fn(&StringRef::empty))) {
      if (const FunctionDecl *Def = Callee->getDefinition()) {
        ParameterNames = getParameterNamesForDecl(Def, ArgCount);
      }
    }
    assert(ParameterNames.size() == ArgCount);

    // Standard library functions often have parameter names that start
    // with underscores, which makes the hints noisy, so strip them out.
    for (auto &Name : ParameterNames)
      stripLeadingUnderscores(Name);

    return ParameterNames;
  }

  static void stripLeadingUnderscores(StringRef &Name) {
    Name = Name.ltrim('_');
  }

  // Return the number of fixed parameters Function has, that is, not counting
  // parameters that are variadic (instantiated from a parameter pack) or
  // C-style varargs.
  static size_t getFixedParamCount(const FunctionDecl *Function) {
    if (FunctionTemplateDecl *Template = Function->getPrimaryTemplate()) {
      FunctionDecl *F = Template->getTemplatedDecl();
      size_t Result = 0;
      for (ParmVarDecl *Parm : F->parameters()) {
        if (Parm->isParameterPack()) {
          break;
        }
        ++Result;
      }
      return Result;
    }
    // C-style varargs don't need special handling, they're already
    // not included in getNumParams().
    return Function->getNumParams();
  }

  static StringRef getSimpleName(const NamedDecl &D) {
    if (IdentifierInfo *Ident = D.getDeclName().getAsIdentifierInfo()) {
      return Ident->getName();
    }

    return StringRef();
  }

  NameVec getParameterNamesForDecl(const FunctionDecl *Function,
                                   size_t ArgCount) {
    NameVec Result;
    for (size_t I = 0; I < ArgCount; ++I) {
      const ParmVarDecl *Parm = Function->getParamDecl(I);
      assert(Parm);
      Result.emplace_back(getSimpleName(*Parm));
    }
    return Result;
  }

  void addInlayHint(SourceRange R, InlayHintKind Kind, llvm::StringRef Label) {
    auto FileRange =
        toHalfOpenFileRange(AST.getSourceManager(), AST.getLangOpts(), R);
    if (!FileRange)
      return;
    Results.push_back(InlayHint{
        Range{
            sourceLocToPosition(AST.getSourceManager(), FileRange->getBegin()),
            sourceLocToPosition(AST.getSourceManager(), FileRange->getEnd())},
        Kind, Label.str()});
  }

  std::vector<InlayHint> &Results;
  ASTContext &AST;
};

std::vector<InlayHint> inlayHints(ParsedAST &AST) {
  std::vector<InlayHint> Results;
  InlayHintVisitor Visitor(Results, AST);
  Visitor.TraverseAST(AST.getASTContext());
  return Results;
}

} // namespace clangd
} // namespace clang
