//===-- LoopConvert/LoopMatchers.cpp - Matchers for for loops -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains definitions of the matchers for use in migrating
/// C++ for loops.
///
//===----------------------------------------------------------------------===//

#include "LoopMatchers.h"

using namespace clang::ast_matchers;
using namespace clang;

const char LoopName[] = "forLoop";
const char ConditionBoundName[] = "conditionBound";
const char ConditionVarName[] = "conditionVar";
const char IncrementVarName[] = "incrementVar";
const char InitVarName[] = "initVar";
const char BeginCallName[] = "beginCall";
const char EndCallName[] = "endCall";
const char ConditionEndVarName[] = "conditionEndVar";
const char EndVarName[] = "endVar";
const char DerefByValueResultName[] = "derefByValueResult";
const char DerefByRefResultName[] = "derefByRefResult";

// shared matchers
static const TypeMatcher AnyType = anything();

static const StatementMatcher IntegerComparisonMatcher =
    expr(ignoringParenImpCasts(declRefExpr(to(
        varDecl(hasType(isInteger())).bind(ConditionVarName)))));

static const DeclarationMatcher InitToZeroMatcher =
    varDecl(hasInitializer(ignoringParenImpCasts(
        integerLiteral(equals(0))))).bind(InitVarName);

static const StatementMatcher IncrementVarMatcher =
    declRefExpr(to(
        varDecl(hasType(isInteger())).bind(IncrementVarName)));

// FIXME: How best to document complicated matcher expressions? They're fairly
// self-documenting...but there may be some unintuitive parts.

/// \brief The matcher for loops over arrays.
///
/// In this general example, assuming 'j' and 'k' are of integral type:
/// \code
///   for (int i = 0; j < 3 + 2; ++k) { ... }
/// \endcode
/// The following string identifiers are bound to these parts of the AST:
///   ConditionVarName: 'j' (as a VarDecl)
///   ConditionBoundName: '3 + 2' (as an Expr)
///   InitVarName: 'i' (as a VarDecl)
///   IncrementVarName: 'k' (as a VarDecl)
///   LoopName: The entire for loop (as a ForStmt)
///
/// Client code will need to make sure that:
///   - The three index variables identified by the matcher are the same
///     VarDecl.
///   - The index variable is only used as an array index.
///   - All arrays indexed by the loop are the same.
StatementMatcher makeArrayLoopMatcher() {
  StatementMatcher ArrayBoundMatcher =
      expr(hasType(isInteger())).bind(ConditionBoundName);

  return forStmt(
      hasLoopInit(declStmt(hasSingleDecl(InitToZeroMatcher))),
      hasCondition(anyOf(binaryOperator(hasOperatorName("<"),
                                        hasLHS(IntegerComparisonMatcher),
                                        hasRHS(ArrayBoundMatcher)),
                         binaryOperator(hasOperatorName(">"),
                                        hasLHS(ArrayBoundMatcher),
                                        hasRHS(IntegerComparisonMatcher)))),
      hasIncrement(unaryOperator(hasOperatorName("++"),
                                 hasUnaryOperand(IncrementVarMatcher))))
      .bind(LoopName);
}

/// \brief The matcher used for iterator-based for loops.
///
/// This matcher is more flexible than array-based loops. It will match
/// catch loops of the following textual forms (regardless of whether the
/// iterator type is actually a pointer type or a class type):
///
/// Assuming f, g, and h are of type containerType::iterator,
/// \code
///   for (containerType::iterator it = container.begin(),
///        e = createIterator(); f != g; ++h) { ... }
///   for (containerType::iterator it = container.begin();
///        f != anotherContainer.end(); ++h) { ... }
/// \endcode
/// The following string identifiers are bound to the parts of the AST:
///   InitVarName: 'it' (as a VarDecl)
///   ConditionVarName: 'f' (as a VarDecl)
///   LoopName: The entire for loop (as a ForStmt)
///   In the first example only:
///     EndVarName: 'e' (as a VarDecl)
///     ConditionEndVarName: 'g' (as a VarDecl)
///   In the second example only:
///     EndCallName: 'container.end()' (as a CXXMemberCallExpr)
///
/// Client code will need to make sure that:
///   - The iterator variables 'it', 'f', and 'h' are the same
///   - The two containers on which 'begin' and 'end' are called are the same
///   - If the end iterator variable 'g' is defined, it is the same as 'f'
StatementMatcher makeIteratorLoopMatcher() {
  StatementMatcher BeginCallMatcher =
      memberCallExpr(
        argumentCountIs(0),
        callee(
          methodDecl(hasName("begin"))
        )
      ).bind(BeginCallName);

  DeclarationMatcher InitDeclMatcher =
      varDecl(
        hasInitializer(
          anyOf(
            ignoringParenImpCasts(BeginCallMatcher),
            materializeTemporaryExpr(ignoringParenImpCasts(BeginCallMatcher)),
            hasDescendant(BeginCallMatcher)
          )
        )
      ).bind(InitVarName);

  DeclarationMatcher EndDeclMatcher =
      varDecl(hasInitializer(anything())).bind(EndVarName);

  StatementMatcher EndCallMatcher =
      memberCallExpr(argumentCountIs(0), callee(methodDecl(hasName("end"))));

  StatementMatcher IteratorBoundMatcher =
      expr(anyOf(ignoringParenImpCasts(declRefExpr(to(
          varDecl().bind(ConditionEndVarName)))),
                 ignoringParenImpCasts(
                     expr(EndCallMatcher).bind(EndCallName)),
                 materializeTemporaryExpr(ignoringParenImpCasts(
                     expr(EndCallMatcher).bind(EndCallName)))));

  StatementMatcher IteratorComparisonMatcher =
      expr(ignoringParenImpCasts(declRefExpr(to(
          varDecl().bind(ConditionVarName)))));

  StatementMatcher OverloadedNEQMatcher = operatorCallExpr(
      hasOverloadedOperatorName("!="),
      argumentCountIs(2),
      hasArgument(0, IteratorComparisonMatcher),
      hasArgument(1, IteratorBoundMatcher));

  // This matcher tests that a declaration is a CXXRecordDecl that has an
  // overloaded operator*(). If the operator*() returns by value instead of by
  // reference then the return type is tagged with DerefByValueResultName.
  internal::Matcher<VarDecl> TestDerefReturnsByValue =
      hasType(
        recordDecl(
          hasMethod(
            allOf(
              hasOverloadedOperatorName("*"),
              anyOf(
                // Tag the return type if it's by value.
                returns(
                  qualType(
                    unless(hasCanonicalType(referenceType()))
                  ).bind(DerefByValueResultName)
                ),
                returns(
                  // Skip loops where the iterator's operator* returns an
                  // rvalue reference. This is just weird.
                  qualType(
                    unless(
                      hasCanonicalType(rValueReferenceType())
                    )
                  ).bind(DerefByRefResultName)
                )
              )
            )
          )
        )
      );


  return
    forStmt(
      hasLoopInit(anyOf(
        declStmt(
          declCountIs(2),
          containsDeclaration(0, InitDeclMatcher),
          containsDeclaration(1, EndDeclMatcher)
        ),
        declStmt(hasSingleDecl(InitDeclMatcher))
      )),
      hasCondition(anyOf(
        binaryOperator(
          hasOperatorName("!="),
          hasLHS(IteratorComparisonMatcher),
          hasRHS(IteratorBoundMatcher)
        ),
        binaryOperator(
          hasOperatorName("!="),
          hasLHS(IteratorBoundMatcher),
          hasRHS(IteratorComparisonMatcher)
        ),
        OverloadedNEQMatcher
      )),
      hasIncrement(anyOf(
        unaryOperator(
          hasOperatorName("++"),
          hasUnaryOperand(
            declRefExpr(to(
              varDecl(hasType(pointsTo(AnyType))).bind(IncrementVarName)
            ))
          )
        ),
        operatorCallExpr(
          hasOverloadedOperatorName("++"),
          hasArgument(0,
            declRefExpr(to(
              varDecl(TestDerefReturnsByValue).bind(IncrementVarName)
            ))
          )
        )
      ))
    ).bind(LoopName);
}

/// \brief The matcher used for array-like containers (pseudoarrays).
///
/// This matcher is more flexible than array-based loops. It will match
/// loops of the following textual forms (regardless of whether the
/// iterator type is actually a pointer type or a class type):
///
/// Assuming f, g, and h are of type containerType::iterator,
/// \code
///   for (int i = 0, j = container.size(); f < g; ++h) { ... }
///   for (int i = 0; f < container.size(); ++h) { ... }
/// \endcode
/// The following string identifiers are bound to the parts of the AST:
///   InitVarName: 'i' (as a VarDecl)
///   ConditionVarName: 'f' (as a VarDecl)
///   LoopName: The entire for loop (as a ForStmt)
///   In the first example only:
///     EndVarName: 'j' (as a VarDecl)
///     ConditionEndVarName: 'g' (as a VarDecl)
///   In the second example only:
///     EndCallName: 'container.size()' (as a CXXMemberCallExpr)
///
/// Client code will need to make sure that:
///   - The index variables 'i', 'f', and 'h' are the same
///   - The containers on which 'size()' is called is the container indexed
///   - The index variable is only used in overloaded operator[] or
///     container.at()
///   - If the end iterator variable 'g' is defined, it is the same as 'j'
///   - The container's iterators would not be invalidated during the loop
StatementMatcher makePseudoArrayLoopMatcher() {
  // Test that the incoming type has a record declaration that has methods
  // called 'begin' and 'end'. If the incoming type is const, then make sure
  // these methods are also marked const.
  //
  // FIXME: To be completely thorough this matcher should also ensure the
  // return type of begin/end is an iterator that dereferences to the same as
  // what operator[] or at() returns. Such a test isn't likely to fail except
  // for pathological cases.
  //
  // FIXME: Also, a record doesn't necessarily need begin() and end(). Free
  // functions called begin() and end() taking the container as an argument
  // are also allowed.
  TypeMatcher RecordWithBeginEnd = 
    qualType(anyOf(
      qualType(
        isConstQualified(),
        hasDeclaration(
          recordDecl(
            hasMethod(
              methodDecl(
                hasName("begin"),
                isConst()
              )
            ),
            hasMethod(
              methodDecl(
                hasName("end"),
                isConst()
              )
            )
          )
        ) // hasDeclaration
      ), // qualType
      qualType(
        unless(isConstQualified()),
        hasDeclaration(
          recordDecl(
            hasMethod(hasName("begin")),
            hasMethod(hasName("end"))
          )
        )
      ) // qualType
    )
  );

  StatementMatcher SizeCallMatcher =
      memberCallExpr(argumentCountIs(0),
                     callee(methodDecl(anyOf(hasName("size"),
                                             hasName("length")))),
                     on(anyOf(hasType(pointsTo(RecordWithBeginEnd)),
                              hasType(RecordWithBeginEnd))));

  StatementMatcher EndInitMatcher =
      expr(anyOf(
          ignoringParenImpCasts(expr(SizeCallMatcher).bind(EndCallName)),
          explicitCastExpr(hasSourceExpression(ignoringParenImpCasts(
              expr(SizeCallMatcher).bind(EndCallName))))));

  DeclarationMatcher EndDeclMatcher =
      varDecl(hasInitializer(EndInitMatcher)).bind(EndVarName);

  StatementMatcher IndexBoundMatcher =
      expr(anyOf(
          ignoringParenImpCasts(declRefExpr(to(
              varDecl(hasType(isInteger())).bind(ConditionEndVarName)))),
          EndInitMatcher));

  return forStmt(
      hasLoopInit(anyOf(
          declStmt(declCountIs(2),
                   containsDeclaration(0, InitToZeroMatcher),
                   containsDeclaration(1, EndDeclMatcher)),
          declStmt(hasSingleDecl(InitToZeroMatcher)))),
      hasCondition(anyOf(
          binaryOperator(hasOperatorName("<"),
                         hasLHS(IntegerComparisonMatcher),
                         hasRHS(IndexBoundMatcher)),
          binaryOperator(hasOperatorName(">"),
                         hasLHS(IndexBoundMatcher),
                         hasRHS(IntegerComparisonMatcher)))),
      hasIncrement(unaryOperator(
          hasOperatorName("++"),
          hasUnaryOperand(IncrementVarMatcher))))
      .bind(LoopName);
}
