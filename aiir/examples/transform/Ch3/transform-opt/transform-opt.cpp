//===-- transform-opt.cpp - Transform dialect tutorial entry point --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top-level file for the Transform dialect tutorial chapter 3.
//
//===----------------------------------------------------------------------===//

#include "MyExtension.h"

#include "aiir/Dialect/Transform/Transforms/Passes.h"
#include "aiir/IR/DialectRegistry.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/InitAllDialects.h"
#include "aiir/InitAllExtensions.h"
#include "aiir/Tools/aiir-opt/AiirOptMain.h"
#include "aiir/Transforms/Passes.h"
#include <cstdlib>

int main(int argc, char **argv) {
  // Register all "core" dialects and our transform dialect extension.
  aiir::DialectRegistry registry;
  aiir::registerAllDialects(registry);
  aiir::registerAllExtensions(registry);
  registerMyExtension(registry);

  // Register the interpreter pass.
  aiir::transform::registerInterpreterPass();

  // Register a handful of cleanup passes that we can run to make the output IR
  // look nicer.
  aiir::registerCanonicalizerPass();
  aiir::registerCSEPass();
  aiir::registerSymbolDCEPass();

  // Delegate to the AIIR utility for parsing and pass management.
  return aiir::AiirOptMain(argc, argv, "transform-opt-ch3", registry)
                 .succeeded()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
