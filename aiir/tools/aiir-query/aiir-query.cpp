//===- aiir-query.cpp - AIIR Query Driver ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that queries a file from/to AIIR using one
// of the registered queries.
//
//===----------------------------------------------------------------------===//

#include "aiir/IR/Dialect.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/Matchers.h"
#include "aiir/InitAllDialects.h"
#include "aiir/Query/Matcher/Registry.h"
#include "aiir/Query/Matcher/SliceMatchers.h"
#include "aiir/Tools/aiir-query/AiirQueryMain.h"

using namespace aiir;

// This is needed because these matchers are defined as overloaded functions.
using HasOpAttrName = detail::AttrOpMatcher(StringRef);
using HasOpName = detail::NameOpMatcher(StringRef);
using IsConstantOp = detail::constant_op_matcher();

namespace test {
#ifdef AIIR_INCLUDE_TESTS
void registerTestDialect(DialectRegistry &);
#endif
} // namespace test

int main(int argc, char **argv) {

  DialectRegistry dialectRegistry;
  registerAllDialects(dialectRegistry);

  query::matcher::Registry matcherRegistry;

  // Matchers registered in alphabetical order for consistency:
  matcherRegistry.registerMatcher("allOf", query::matcher::internal::allOf);
  matcherRegistry.registerMatcher("anyOf", query::matcher::internal::anyOf);
  matcherRegistry.registerMatcher(
      "getAllDefinitions",
      query::matcher::m_GetAllDefinitions<query::matcher::DynMatcher>);
  matcherRegistry.registerMatcher(
      "getDefinitions",
      query::matcher::m_GetDefinitions<query::matcher::DynMatcher>);
  matcherRegistry.registerMatcher(
      "getDefinitionsByPredicate",
      query::matcher::m_GetDefinitionsByPredicate<query::matcher::DynMatcher,
                                                  query::matcher::DynMatcher>);
  matcherRegistry.registerMatcher(
      "getUsersByPredicate",
      query::matcher::m_GetUsersByPredicate<query::matcher::DynMatcher,
                                            query::matcher::DynMatcher>);
  matcherRegistry.registerMatcher("hasOpAttrName",
                                  static_cast<HasOpAttrName *>(m_Attr));
  matcherRegistry.registerMatcher("hasOpName", static_cast<HasOpName *>(m_Op));
  matcherRegistry.registerMatcher("isConstantOp",
                                  static_cast<IsConstantOp *>(m_Constant));
  matcherRegistry.registerMatcher("isNegInfFloat", m_NegInfFloat);
  matcherRegistry.registerMatcher("isNegZeroFloat", m_NegZeroFloat);
  matcherRegistry.registerMatcher("isNonZero", m_NonZero);
  matcherRegistry.registerMatcher("isOne", m_One);
  matcherRegistry.registerMatcher("isOneFloat", m_OneFloat);
  matcherRegistry.registerMatcher("isPosInfFloat", m_PosInfFloat);
  matcherRegistry.registerMatcher("isPosZeroFloat", m_PosZeroFloat);
  matcherRegistry.registerMatcher("isZero", m_Zero);
  matcherRegistry.registerMatcher("isZeroFloat", m_AnyZeroFloat);

#ifdef AIIR_INCLUDE_TESTS
  test::registerTestDialect(dialectRegistry);
#endif
  AIIRContext context(dialectRegistry);

  return failed(aiirQueryMain(argc, argv, context, matcherRegistry));
}
