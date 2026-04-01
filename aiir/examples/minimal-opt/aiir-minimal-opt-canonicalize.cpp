//===- aiir-minimal-opt-canonicalize.cpp ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Tools/aiir-opt/AiirOptMain.h"
#include "aiir/Transforms/Passes.h"

int main(int argc, char **argv) {
  // Register only the canonicalize pass
  // This pulls in the pattern rewrite engine as well as the whole PDL
  // compiler/intepreter.
  aiir::registerCanonicalizerPass();

  aiir::DialectRegistry registry;
  return aiir::asMainReturnCode(aiir::AiirOptMain(
      argc, argv, "Minimal Standalone optimizer driver\n", registry));
}
