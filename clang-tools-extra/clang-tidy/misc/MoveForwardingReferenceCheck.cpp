//===--- MoveForwardingReferenceCheck.cpp - clang-tidy --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MoveForwardingReferenceCheck.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

static void replaceMoveWithForward(const UnresolvedLookupExpr *Callee,
                                   const ParmVarDecl *ParmVar,
                                   const TemplateTypeParmDecl *TypeParmDecl,
                                   DiagnosticBuilder &Diag,
                                   const ASTContext &Context) {
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  CharSourceRange CallRange =
      Lexer::makeFileCharRange(CharSourceRange::getTokenRange(
                                   Callee->getLocStart(), Callee->getLocEnd()),
                               SM, LangOpts);

  if (CallRange.isValid()) {
    const std::string TypeName =
        TypeParmDecl->getIdentifier()
            ? TypeParmDecl->getName().str()
            : (llvm::Twine("decltype(") + ParmVar->getName() + ")").str();

    const std::string ForwardName =
        (llvm::Twine("forward<") + TypeName + ">").str();

    // Create a replacement only if we see a "standard" way of calling
    // std::move(). This will hopefully prevent erroneous replacements if the
    // code does unusual things (e.g. create an alias for std::move() in
    // another namespace).
    NestedNameSpecifier *NNS = Callee->getQualifier();
    if (!NNS) {
      // Called as "move" (i.e. presumably the code had a "using std::move;").
      // We still conservatively put a "std::" in front of the forward because
      // we don't know whether the code also had a "using std::forward;".
      Diag << FixItHint::CreateReplacement(CallRange, "std::" + ForwardName);
    } else if (const NamespaceDecl *Namespace = NNS->getAsNamespace()) {
      if (Namespace->getName() == "std") {
        if (!NNS->getPrefix()) {
          // Called as "std::move".
          Diag << FixItHint::CreateReplacement(CallRange,
                                               "std::" + ForwardName);
        } else if (NNS->getPrefix()->getKind() == NestedNameSpecifier::Global) {
          // Called as "::std::move".
          Diag << FixItHint::CreateReplacement(CallRange,
                                               "::std::" + ForwardName);
        }
      }
    }
  }
}

void MoveForwardingReferenceCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus11)
    return;

  // Matches a ParmVarDecl for a forwarding reference, i.e. a non-const rvalue
  // reference of a function template parameter type.
  auto ForwardingReferenceParmMatcher =
      parmVarDecl(
          hasType(qualType(rValueReferenceType(),
                           references(templateTypeParmType(hasDeclaration(
                               templateTypeParmDecl().bind("type-parm-decl")))),
                           unless(references(qualType(isConstQualified()))))))
          .bind("parm-var");

  Finder->addMatcher(
      callExpr(callee(unresolvedLookupExpr(
                          hasAnyDeclaration(namedDecl(
                              hasUnderlyingDecl(hasName("::std::move")))))
                          .bind("lookup")),
               argumentCountIs(1),
               hasArgument(0, ignoringParenImpCasts(declRefExpr(
                                  to(ForwardingReferenceParmMatcher)))))
          .bind("call-move"),
      this);
}

void MoveForwardingReferenceCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *CallMove = Result.Nodes.getNodeAs<CallExpr>("call-move");
  const auto *UnresolvedLookup =
      Result.Nodes.getNodeAs<UnresolvedLookupExpr>("lookup");
  const auto *ParmVar = Result.Nodes.getNodeAs<ParmVarDecl>("parm-var");
  const auto *TypeParmDecl =
      Result.Nodes.getNodeAs<TemplateTypeParmDecl>("type-parm-decl");

  // Get the FunctionDecl and FunctionTemplateDecl containing the function
  // parameter.
  const FunctionDecl *FuncForParam =
      dyn_cast<FunctionDecl>(ParmVar->getDeclContext());
  if (!FuncForParam)
    return;
  const FunctionTemplateDecl *FuncTemplate =
      FuncForParam->getDescribedFunctionTemplate();
  if (!FuncTemplate)
    return;

  // Check that the template type parameter belongs to the same function
  // template as the function parameter of that type. (This implies that type
  // deduction will happen on the type.)
  const TemplateParameterList *Params = FuncTemplate->getTemplateParameters();
  if (!std::count(Params->begin(), Params->end(), TypeParmDecl))
    return;

  auto Diag = diag(CallMove->getExprLoc(),
                   "forwarding reference passed to std::move(), which may "
                   "unexpectedly cause lvalues to be moved; use "
                   "std::forward() instead");

  replaceMoveWithForward(UnresolvedLookup, ParmVar, TypeParmDecl, Diag,
                         *Result.Context);
}

} // namespace misc
} // namespace tidy
} // namespace clang
