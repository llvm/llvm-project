//===-- nullptr-convert/Matchers.cpp - Matchers for null casts ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the definitions for matcher-generating functions
///  and a custom AST_MATCHER for identifying casts of type CK_NullTo*.
///
//===----------------------------------------------------------------------===//
#include "NullptrMatchers.h"
#include "clang/AST/ASTContext.h"

using namespace clang::ast_matchers;
using namespace clang;

const char *CastSequence = "sequence";

namespace clang {
namespace ast_matchers {
/// \brief Matches cast expressions that have a cast kind of CK_NullToPointer
/// or CK_NullToMemberPointer.
///
/// Given
/// \code
///   int *p = 0;
/// \endcode
/// implicitCastExpr(isNullToPointer()) matches the implicit cast clang adds
/// around \c 0.
AST_MATCHER(CastExpr, isNullToPointer) {
  return Node.getCastKind() == CK_NullToPointer ||
    Node.getCastKind() == CK_NullToMemberPointer;
}

AST_MATCHER(Type, sugaredNullptrType) {
  const Type *DesugaredType = Node.getUnqualifiedDesugaredType();
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(DesugaredType))
    return BT->getKind() == BuiltinType::NullPtr;
  return false;
}

} // end namespace ast_matchers
} // end namespace clang

StatementMatcher makeCastSequenceMatcher() {
  StatementMatcher ImplicitCastToNull =
    implicitCastExpr(
      isNullToPointer(),
      unless(
        hasSourceExpression(
          hasType(sugaredNullptrType())
        )
      )
    );

  return castExpr(
           anyOf(
             ImplicitCastToNull,
             explicitCastExpr(
               hasDescendant(ImplicitCastToNull)
             )
           ),
           unless(hasAncestor(explicitCastExpr()))
         ).bind(CastSequence);
}
