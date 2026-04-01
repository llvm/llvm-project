//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/InitAllDialects.h"
#include "aiir/InitAllPasses.h"
#include "aiir/Support/FileUtilities.h"
#include "aiir/Tools/aiir-opt/AiirOptMain.h"

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandalonePasses.h"

int main(int argc, char **argv) {
  aiir::registerAllPasses();
  aiir::standalone::registerPasses();
  // TODO: Register standalone passes here.

  aiir::DialectRegistry registry;
  registry.insert<aiir::standalone::StandaloneDialect,
                  aiir::arith::ArithDialect, aiir::func::FuncDialect>();
  // Add the following to include *all* AIIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return aiir::asMainReturnCode(
      aiir::AiirOptMain(argc, argv, "Standalone optimizer driver\n", registry));
}
