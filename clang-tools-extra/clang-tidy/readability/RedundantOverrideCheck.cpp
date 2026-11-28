//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantOverrideCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static bool isForwardingType(QualType Type) {
  return Type->isVoidType() || Type->isReferenceType() ||
         (Type->isScalarType() && !Type.isVolatileQualified());
}

static bool hasAdditionalInterface(const CXXMethodDecl *Method) {
  for (const FunctionDecl *Redecl : Method->redecls()) {
    if (Redecl->getBeginLoc().isMacroID() || Redecl->getEndLoc().isMacroID())
      return true;
    if (llvm::any_of(Redecl->attrs(),
                     [](const Attr *A) { return !isa<OverrideAttr>(A); }))
      return true;
    if (llvm::any_of(Redecl->parameters(), [](const ParmVarDecl *P) {
          return P->hasDefaultArg() || P->hasAttrs() ||
                 !isForwardingType(P->getType());
        }))
      return true;
  }
  return false;
}

static bool hasOtherOverloads(const CXXRecordDecl *Record,
                              const CXXMethodDecl *Method) {
  return llvm::any_of(
      Record->lookup(Method->getDeclName()), [Method](const NamedDecl *D) {
        return D->getCanonicalDecl() != Method->getCanonicalDecl();
      });
}

static const CXXMethodDecl *getRedundantBase(const CXXMethodDecl *Method) {
  const CXXRecordDecl *Record = Method->getParent();
  if (Method->size_overridden_methods() != 1 || Method->isPureVirtual() ||
      !Method->getDeclName().isIdentifier() || Record->isDependentContext() ||
      isa<ClassTemplateSpecializationDecl>(Record) ||
      Method->getTemplatedKind() != FunctionDecl::TK_NonTemplate ||
      Record->getNumBases() != 1 || Record->hasAttrs() ||
      hasAdditionalInterface(Method))
    return nullptr;

  const CXXMethodDecl *Base = *Method->begin_overridden_methods();
  const CXXBaseSpecifier &Inheritance = *Record->bases_begin();
  const CXXRecordDecl *BaseRecord = Inheritance.getType()->getAsCXXRecordDecl();
  if (!BaseRecord || Inheritance.isVirtual() || Base->isPureVirtual() ||
      Base->getAccess() != AS_public ||
      isa<ClassTemplateSpecializationDecl>(BaseRecord) ||
      BaseRecord->hasAttrs() ||
      BaseRecord->getCanonicalDecl() != Base->getParent()->getCanonicalDecl() ||
      hasAdditionalInterface(Base) || hasOtherOverloads(Record, Method) ||
      hasOtherOverloads(BaseRecord, Base))
    return nullptr;

  // Removing the override must preserve access through the derived class,
  // including access restricted by protected or private inheritance.
  const AccessSpecifier InheritedAccess =
      std::max(Base->getAccess(), Inheritance.getAccessSpecifier());
  if (InheritedAccess != AS_public || Method->getAccess() != InheritedAccess ||
      !ASTContext::hasSameType(Method->getType(), Base->getType()) ||
      !isForwardingType(Method->getReturnType()) ||
      Method->isConstexpr() != Base->isConstexpr() ||
      Method->isConsteval() != Base->isConsteval())
    return nullptr;

  const auto *MethodType = Method->getType()->castAs<FunctionProtoType>();
  const auto *BaseType = Base->getType()->castAs<FunctionProtoType>();
  if (MethodType->isVariadic() || MethodType->hasDynamicExceptionSpec() ||
      MethodType->getExceptionSpecType() != BaseType->getExceptionSpecType() ||
      isUnresolvedExceptionSpec(MethodType->getExceptionSpecType()) ||
      MethodType->canThrow() == CT_Dependent ||
      MethodType->canThrow() != BaseType->canThrow())
    return nullptr;

  const auto *Body = dyn_cast<CompoundStmt>(Method->getBody());
  if (!Body || Body->size() != 1)
    return nullptr;
  const Stmt *Statement = Body->body_front();
  const Expr *Expression = nullptr;
  if (const auto *Return = dyn_cast<ReturnStmt>(Statement))
    Expression = Return->getRetValue();
  else if (Method->getReturnType()->isVoidType())
    Expression = dyn_cast<Expr>(Statement);
  if (!Expression)
    return nullptr;

  const auto *Call =
      dyn_cast<CXXMemberCallExpr>(Expression->IgnoreParenImpCasts());
  if (!Call || !Call->getMethodDecl() || Call->getExprLoc().isMacroID() ||
      Call->getMethodDecl()->getCanonicalDecl() != Base->getCanonicalDecl() ||
      Call->getNumArgs() != Method->getNumParams())
    return nullptr;
  const auto *Member =
      dyn_cast<MemberExpr>(Call->getCallee()->IgnoreParenImpCasts());
  if (!Member || !Member->hasQualifier() ||
      !isa<CXXThisExpr>(
          Call->getImplicitObjectArgument()->IgnoreParenImpCasts()))
    return nullptr;
  for (unsigned I = 0; I != Call->getNumArgs(); ++I) {
    const auto *Argument =
        dyn_cast<DeclRefExpr>(Call->getArg(I)->IgnoreParenImpCasts());
    if (!Argument || Argument->getLocation().isMacroID() ||
        Argument->getDecl() != Method->getParamDecl(I))
      return nullptr;
  }
  return Base;
}

static bool getRemovalRanges(const CXXMethodDecl *Method,
                             const SourceManager &SM,
                             const LangOptions &LangOpts,
                             SmallVectorImpl<CharSourceRange> &Ranges) {
  for (const FunctionDecl *Redecl : Method->redecls()) {
    const SourceRange Range = Redecl->getSourceRange();
    if (Range.isInvalid() || Range.getBegin().isMacroID() ||
        Range.getEnd().isMacroID() || SM.isInSystemHeader(Range.getBegin()))
      return false;
    // A shared declaration such as "void f(), g();" cannot be removed whole.
    for (const Decl *Other : Redecl->getLexicalDeclContext()->decls())
      if (Other->getCanonicalDecl() != Redecl->getCanonicalDecl() &&
          Other->getBeginLoc() == Range.getBegin())
        return false;

    CharSourceRange FileRange = Lexer::makeFileCharRange(
        CharSourceRange::getTokenRange(Range), SM, LangOpts);
    if (FileRange.isInvalid())
      return false;
    if (!Redecl->doesThisDeclarationHaveABody()) {
      const SourceLocation AfterSemicolon = Lexer::findLocationAfterToken(
          Range.getEnd(), tok::semi, SM, LangOpts,
          /*SkipTrailingWhitespaceAndNewLine=*/false);
      if (AfterSemicolon.isInvalid())
        return false;
      FileRange.setEnd(AfterSemicolon);
    }
    SourceLocation Loc = FileRange.getBegin();
    while (Loc < FileRange.getEnd()) {
      Token Tok;
      if (Lexer::getRawToken(Loc, Tok, SM, LangOpts,
                             /*IgnoreWhiteSpace=*/true))
        return false;
      if (Tok.getLocation() >= FileRange.getEnd())
        break;
      if (Tok.isOneOf(tok::hash, tok::eof))
        return false;
      Loc = Tok.getEndLoc();
    }
    Ranges.push_back(FileRange);
  }
  return true;
}

bool RedundantOverrideCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}

std::optional<TraversalKind>
RedundantOverrideCheck::getCheckTraversalKind() const {
  return TK_IgnoreUnlessSpelledInSource;
}

void RedundantOverrideCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(cxxMethodDecl(isDefinition(), isVirtual(),
                                   unless(cxxDestructorDecl()),
                                   hasBody(compoundStmt(statementCountIs(1))))
                         .bind("method"),
                     this);
  Finder->addMatcher(declRefExpr(to(cxxMethodDecl().bind("referenced-method"))),
                     this);
}

void RedundantOverrideCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Reference =
          Result.Nodes.getNodeAs<CXXMethodDecl>("referenced-method")) {
    // A pointer-to-member or its type can depend on the derived declaration.
    ReferencedMethods.insert(Reference->getCanonicalDecl());
    return;
  }
  const auto *Method = Result.Nodes.getNodeAs<CXXMethodDecl>("method");
  if (getRedundantBase(Method))
    Candidates.push_back(Method);
}

void RedundantOverrideCheck::onEndOfTranslationUnit() {
  for (const CXXMethodDecl *Method : Candidates) {
    if (ReferencedMethods.contains(Method->getCanonicalDecl()) ||
        Method->getLocation().isMacroID())
      continue;
    SmallVector<CharSourceRange, 2> Ranges;
    if (!getRemovalRanges(Method, Method->getASTContext().getSourceManager(),
                          getLangOpts(), Ranges))
      continue;
    const auto Diag = diag(Method->getLocation(), "redundant override of %0")
                      << *Method->begin_overridden_methods();
    for (const CharSourceRange Range : Ranges)
      Diag << FixItHint::CreateRemoval(Range);
  }
  Candidates.clear();
  ReferencedMethods.clear();
}

} // namespace clang::tidy::readability
