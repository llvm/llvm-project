//===--- FoldInitTypeCheck.cpp - clang-tidy--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FoldInitTypeCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void FoldInitTypeCheck::registerMatchers(MatchFinder *Finder) {
  // We match functions of interest and bind the iterator and init value types.
  // Note: Right now we check only builtin types.
  const auto BuiltinTypeWithId = [](const char *ID) {
    return hasCanonicalType(builtinType().bind(ID));
  };
  const auto IteratorWithValueType = [&BuiltinTypeWithId](const char *ID) {
    return anyOf(
        // Pointer types.
        pointsTo(BuiltinTypeWithId(ID)),
        // Iterator types.
        recordType(hasDeclaration(has(typedefNameDecl(
            hasName("value_type"), hasType(BuiltinTypeWithId(ID)))))));
  };

  const auto IteratorParam = parmVarDecl(
      hasType(hasCanonicalType(IteratorWithValueType("IterValueType"))));
  const auto Iterator2Param = parmVarDecl(
      hasType(hasCanonicalType(IteratorWithValueType("Iter2ValueType"))));
  const auto InitParam = parmVarDecl(hasType(BuiltinTypeWithId("InitType")));

  // std::accumulate, std::reduce.
  Finder->addMatcher(
      callExpr(callee(functionDecl(
                   hasAnyName("::std::accumulate", "::std::reduce"),
                   hasParameter(0, IteratorParam), hasParameter(2, InitParam))),
               argumentCountIs(3))
          .bind("Call"),
      this);
  // std::inner_product.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::std::inner_product"),
                                   hasParameter(0, IteratorParam),
                                   hasParameter(2, Iterator2Param),
                                   hasParameter(3, InitParam))),
               argumentCountIs(4))
          .bind("Call"),
      this);
  // std::reduce with a policy.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::std::reduce"),
                                   hasParameter(1, IteratorParam),
                                   hasParameter(3, InitParam))),
               argumentCountIs(4))
          .bind("Call"),
      this);
  // std::inner_product with a policy.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::std::inner_product"),
                                   hasParameter(1, IteratorParam),
                                   hasParameter(3, Iterator2Param),
                                   hasParameter(4, InitParam))),
               argumentCountIs(5))
          .bind("Call"),
      this);
}

/// Returns true if ValueType is allowed to fold into InitType, i.e. if:
///   static_cast<InitType>(ValueType{some_value})
/// does not result in trucation.
static bool isValidBuiltinFold(const BuiltinType &ValueType,
                               const BuiltinType &InitType,
                               const ASTContext &Context) {
  const auto ValueTypeSize = Context.getTypeSize(&ValueType);
  const auto InitTypeSize = Context.getTypeSize(&InitType);
  // It's OK to fold a float into a float of bigger or equal size, but not OK to
  // fold into an int.
  if (ValueType.isFloatingPoint())
    return InitType.isFloatingPoint() && InitTypeSize >= ValueTypeSize;
  // It's OK to fold an int into:
  //  - an int of the same size and signedness.
  //  - a bigger int, regardless of signedness.
  //  - FIXME: should it be a warning to fold into floating point?
  if (ValueType.isInteger()) {
    if (InitType.isInteger()) {
      if (InitType.isSignedInteger() == ValueType.isSignedInteger())
        return InitTypeSize >= ValueTypeSize;
      return InitTypeSize > ValueTypeSize;
    }
    if (InitType.isFloatingPoint())
      return InitTypeSize >= ValueTypeSize;
  }
  return false;
}

/// Prints a diagnostic if IterValueType doe snot fold into IterValueType (see
// isValidBuiltinFold for details).
void FoldInitTypeCheck::doCheck(const BuiltinType &IterValueType,
                                const BuiltinType &InitType,
                                const ASTContext &Context,
                                const CallExpr &CallNode) {
  if (!isValidBuiltinFold(IterValueType, InitType, Context)) {
    diag(CallNode.getExprLoc(), "folding type %0 into type %1 might result in "
                                "loss of precision")
        << IterValueType.desugar() << InitType.desugar();
  }
}

void FoldInitTypeCheck::check(const MatchFinder::MatchResult &Result) {
  // Given the iterator and init value type retreived by the matchers,
  // we check that the ::value_type of the iterator is compatible with
  // the init value type.
  const auto *InitType = Result.Nodes.getNodeAs<BuiltinType>("InitType");
  const auto *IterValueType =
      Result.Nodes.getNodeAs<BuiltinType>("IterValueType");
  assert(InitType != nullptr);
  assert(IterValueType != nullptr);

  const auto *CallNode = Result.Nodes.getNodeAs<CallExpr>("Call");
  assert(CallNode != nullptr);

  doCheck(*IterValueType, *InitType, *Result.Context, *CallNode);

  if (const auto *Iter2ValueType =
          Result.Nodes.getNodeAs<BuiltinType>("Iter2ValueType"))
    doCheck(*Iter2ValueType, *InitType, *Result.Context, *CallNode);
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
