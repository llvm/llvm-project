//===--- UseEqualsDefaultCheck.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseEqualsDefaultCheck.h"
#include "../utils/LexerUtils.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

static const char SpecialFunction[] = "SpecialFunction";

/// Finds all the named non-static fields of \p Record.
static std::set<const FieldDecl *>
getAllNamedFields(const CXXRecordDecl *Record) {
  std::set<const FieldDecl *> Result;
  for (const auto *Field : Record->fields()) {
    // Static data members are not in this range.
    if (Field->isUnnamedBitfield())
      continue;
    Result.insert(Field);
  }
  return Result;
}

/// Returns the names of the direct bases of \p Record, both virtual and
/// non-virtual.
static std::set<const Type *> getAllDirectBases(const CXXRecordDecl *Record) {
  std::set<const Type *> Result;
  for (auto Base : Record->bases()) {
    // CXXBaseSpecifier.
    const auto *BaseType = Base.getTypeSourceInfo()->getType().getTypePtr();
    Result.insert(BaseType);
  }
  return Result;
}

/// Returns a matcher that matches member expressions where the base is
/// the variable declared as \p Var and the accessed member is the one declared
/// as \p Field.
internal::Matcher<Expr> accessToFieldInVar(const FieldDecl *Field,
                                           const ValueDecl *Var) {
  return ignoringImpCasts(
      memberExpr(hasObjectExpression(declRefExpr(to(varDecl(equalsNode(Var))))),
                 member(fieldDecl(equalsNode(Field)))));
}

/// Check that the given constructor has copy signature and that it
/// copy-initializes all its bases and members.
static bool isCopyConstructorAndCanBeDefaulted(ASTContext *Context,
                                               const CXXConstructorDecl *Ctor) {
  // An explicitly-defaulted constructor cannot have default arguments.
  if (Ctor->getMinRequiredArguments() != 1)
    return false;

  const auto *Record = Ctor->getParent();
  const auto *Param = Ctor->getParamDecl(0);

  // Base classes and members that have to be copied.
  auto BasesToInit = getAllDirectBases(Record);
  auto FieldsToInit = getAllNamedFields(Record);

  // Ensure that all the bases are copied.
  for (const auto *Base : BasesToInit) {
    // The initialization of a base class should be a call to a copy
    // constructor of the base.
    if (match(
            traverse(TK_AsIs,
                     cxxConstructorDecl(
                         forEachConstructorInitializer(cxxCtorInitializer(
                             isBaseInitializer(),
                             withInitializer(cxxConstructExpr(
                                 hasType(equalsNode(Base)),
                                 hasDeclaration(
                                     cxxConstructorDecl(isCopyConstructor())),
                                 argumentCountIs(1),
                                 hasArgument(0, declRefExpr(to(varDecl(
                                                    equalsNode(Param))))))))))),
            *Ctor, *Context)
            .empty())
      return false;
  }

  // Ensure that all the members are copied.
  for (const auto *Field : FieldsToInit) {
    auto AccessToFieldInParam = accessToFieldInVar(Field, Param);
    // The initialization is a CXXConstructExpr for class types.
    if (match(traverse(
                  TK_AsIs,
                  cxxConstructorDecl(
                      forEachConstructorInitializer(cxxCtorInitializer(
                          isMemberInitializer(), forField(equalsNode(Field)),
                          withInitializer(anyOf(
                              AccessToFieldInParam,
                              initListExpr(has(AccessToFieldInParam)),
                              cxxConstructExpr(
                                  hasDeclaration(
                                      cxxConstructorDecl(isCopyConstructor())),
                                  argumentCountIs(1),
                                  hasArgument(0, AccessToFieldInParam)))))))),
              *Ctor, *Context)
            .empty())
      return false;
  }

  // Ensure that we don't do anything else, like initializing an indirect base.
  return Ctor->getNumCtorInitializers() ==
         BasesToInit.size() + FieldsToInit.size();
}

/// Checks that the given method is an overloading of the assignment
/// operator, has copy signature, returns a reference to "*this" and copies
/// all its members and subobjects.
static bool isCopyAssignmentAndCanBeDefaulted(ASTContext *Context,
                                              const CXXMethodDecl *Operator) {
  const auto *Record = Operator->getParent();
  const auto *Param = Operator->getParamDecl(0);

  // Base classes and members that have to be copied.
  auto BasesToInit = getAllDirectBases(Record);
  auto FieldsToInit = getAllNamedFields(Record);

  const auto *Compound = cast<CompoundStmt>(Operator->getBody());

  // The assignment operator definition has to end with the following return
  // statement:
  //   return *this;
  if (Compound->body_empty() ||
      match(traverse(
                TK_AsIs,
                returnStmt(has(ignoringParenImpCasts(unaryOperator(
                    hasOperatorName("*"), hasUnaryOperand(cxxThisExpr())))))),
            *Compound->body_back(), *Context)
          .empty())
    return false;

  // Ensure that all the bases are copied.
  for (const auto *Base : BasesToInit) {
    // Assignment operator of a base class:
    //   Base::operator=(Other);
    //
    // Clang translates this into:
    //   ((Base*)this)->operator=((Base)Other);
    //
    // So we are looking for a member call that fulfills:
    if (match(traverse(
                  TK_AsIs,
                  compoundStmt(has(ignoringParenImpCasts(cxxMemberCallExpr(
                      // - The object is an implicit cast of 'this' to a
                      // pointer to
                      //   a base class.
                      onImplicitObjectArgument(implicitCastExpr(
                          hasImplicitDestinationType(hasCanonicalType(pointsTo(
                              type(equalsNode(Base->getCanonicalTypeInternal()
                                                  .getTypePtr()))))),
                          hasSourceExpression(cxxThisExpr()))),
                      // - The called method is the operator=.
                      callee(cxxMethodDecl(isCopyAssignmentOperator())),
                      // - The argument is (an implicit cast to a Base of)
                      // the argument taken by "Operator".
                      argumentCountIs(1),
                      hasArgument(
                          0, declRefExpr(to(varDecl(equalsNode(Param)))))))))),
              *Compound, *Context)
            .empty())
      return false;
  }

  // Ensure that all the members are copied.
  for (const auto *Field : FieldsToInit) {
    // The assignment of data members:
    //   Field = Other.Field;
    // Is a BinaryOperator in non-class types, and a CXXOperatorCallExpr
    // otherwise.
    auto LHS = memberExpr(hasObjectExpression(cxxThisExpr()),
                          member(fieldDecl(equalsNode(Field))));
    auto RHS = accessToFieldInVar(Field, Param);
    if (match(traverse(TK_AsIs,
                       compoundStmt(has(ignoringParenImpCasts(binaryOperation(
                           hasOperatorName("="), hasLHS(LHS), hasRHS(RHS)))))),
              *Compound, *Context)
            .empty())
      return false;
  }

  // Ensure that we don't do anything else.
  return Compound->size() == BasesToInit.size() + FieldsToInit.size() + 1;
}

/// Returns false if the body has any non-whitespace character.
static bool bodyEmpty(const ASTContext *Context, const CompoundStmt *Body) {
  bool Invalid = false;
  StringRef Text = Lexer::getSourceText(
      CharSourceRange::getCharRange(Body->getLBracLoc().getLocWithOffset(1),
                                    Body->getRBracLoc()),
      Context->getSourceManager(), Context->getLangOpts(), &Invalid);
  return !Invalid && std::strspn(Text.data(), " \t\r\n") == Text.size();
}

UseEqualsDefaultCheck::UseEqualsDefaultCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", true)) {}

void UseEqualsDefaultCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void UseEqualsDefaultCheck::registerMatchers(MatchFinder *Finder) {
  // Skip unions/union-like classes since their constructors behave differently
  // when defaulted vs. empty.
  auto IsUnionLikeClass = recordDecl(
      anyOf(isUnion(),
            has(fieldDecl(isImplicit(), hasType(cxxRecordDecl(isUnion()))))));

  // Destructor.
  Finder->addMatcher(
      cxxDestructorDecl(unless(hasParent(IsUnionLikeClass)), isDefinition())
          .bind(SpecialFunction),
      this);
  Finder->addMatcher(
      cxxConstructorDecl(
          unless(hasParent(IsUnionLikeClass)), isDefinition(),
          anyOf(
              // Default constructor.
              allOf(unless(hasAnyConstructorInitializer(isWritten())),
                    parameterCountIs(0)),
              // Copy constructor.
              allOf(isCopyConstructor(),
                    // Discard constructors that can be used as a copy
                    // constructor because all the other arguments have
                    // default values.
                    parameterCountIs(1))))
          .bind(SpecialFunction),
      this);
  // Copy-assignment operator.
  Finder->addMatcher(
      cxxMethodDecl(unless(hasParent(IsUnionLikeClass)), isDefinition(),
                    isCopyAssignmentOperator(),
                    // isCopyAssignmentOperator() allows the parameter to be
                    // passed by value, and in this case it cannot be
                    // defaulted.
                    hasParameter(0, hasType(lValueReferenceType())),
                    // isCopyAssignmentOperator() allows non lvalue reference
                    // return types, and in this case it cannot be defaulted.
                    returns(qualType(hasCanonicalType(
                        allOf(lValueReferenceType(pointee(type())),
                              unless(matchers::isReferenceToConst()))))))
          .bind(SpecialFunction),
      this);
}

void UseEqualsDefaultCheck::check(const MatchFinder::MatchResult &Result) {
  // Both CXXConstructorDecl and CXXDestructorDecl inherit from CXXMethodDecl.
  const auto *SpecialFunctionDecl =
      Result.Nodes.getNodeAs<CXXMethodDecl>(SpecialFunction);

  if (IgnoreMacros && SpecialFunctionDecl->getLocation().isMacroID())
    return;

  // Discard explicitly deleted/defaulted special member functions and those
  // that are not user-provided (automatically generated).
  if (SpecialFunctionDecl->isDeleted() ||
      SpecialFunctionDecl->isExplicitlyDefaulted() ||
      SpecialFunctionDecl->isLateTemplateParsed() ||
      SpecialFunctionDecl->isTemplateInstantiation() ||
      !SpecialFunctionDecl->isUserProvided() || !SpecialFunctionDecl->hasBody())
    return;

  const auto *Body = dyn_cast<CompoundStmt>(SpecialFunctionDecl->getBody());
  if (!Body)
    return;

  // If there is code inside the body, don't warn.
  if (!SpecialFunctionDecl->isCopyAssignmentOperator() && !Body->body_empty())
    return;

  // If there are comments inside the body, don't do the change.
  bool ApplyFix = SpecialFunctionDecl->isCopyAssignmentOperator() ||
                  bodyEmpty(Result.Context, Body);

  std::vector<FixItHint> RemoveInitializers;
  unsigned MemberType;
  if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(SpecialFunctionDecl)) {
    if (Ctor->getNumParams() == 0) {
      MemberType = 0;
    } else {
      if (!isCopyConstructorAndCanBeDefaulted(Result.Context, Ctor))
        return;
      MemberType = 1;
      // If there are constructor initializers, they must be removed.
      for (const auto *Init : Ctor->inits()) {
        RemoveInitializers.emplace_back(
            FixItHint::CreateRemoval(Init->getSourceRange()));
      }
    }
  } else if (isa<CXXDestructorDecl>(SpecialFunctionDecl)) {
    MemberType = 2;
  } else {
    if (!isCopyAssignmentAndCanBeDefaulted(Result.Context, SpecialFunctionDecl))
      return;
    MemberType = 3;
  }

  // The location of the body is more useful inside a macro as spelling and
  // expansion locations are reported.
  SourceLocation Location = SpecialFunctionDecl->getLocation();
  if (Location.isMacroID())
    Location = Body->getBeginLoc();

  auto Diag = diag(
      Location,
      "use '= default' to define a trivial %select{default constructor|copy "
      "constructor|destructor|copy-assignment operator}0");
  Diag << MemberType;

  if (ApplyFix) {
    // Skipping comments, check for a semicolon after Body->getSourceRange()
    Optional<Token> Token = utils::lexer::findNextTokenSkippingComments(
        Body->getSourceRange().getEnd().getLocWithOffset(1),
        Result.Context->getSourceManager(), Result.Context->getLangOpts());
    StringRef Replacement =
        Token && Token->is(tok::semi) ? "= default" : "= default;";
    Diag << FixItHint::CreateReplacement(Body->getSourceRange(), Replacement)
         << RemoveInitializers;
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
