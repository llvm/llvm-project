//===- aiir-reduce.cpp - The AIIR reducer ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the general framework of the AIIR reducer tool. It
// parses the command line arguments, parses the initial AIIR test case and sets
// up the testing environment. It  outputs the most reduced test case variant
// after executing the reduction passes.
//
//===----------------------------------------------------------------------===//

#include "aiir/IR/Dialect.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/InitAllDialects.h"
#include "aiir/InitAllPasses.h"
#include "aiir/Tools/aiir-reduce/AiirReduceMain.h"

using namespace aiir;

namespace test {
#ifdef AIIR_INCLUDE_TESTS
void registerTestDialect(DialectRegistry &);
#endif
} // namespace test

int main(int argc, char **argv) {
  registerAllPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
#ifdef AIIR_INCLUDE_TESTS
  test::registerTestDialect(registry);
#endif
  AIIRContext context(registry);

  return failed(aiirReduceMain(argc, argv, context));
}
