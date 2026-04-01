//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A language server for ClangIR
//
//===----------------------------------------------------------------------===//

#include "aiir/IR/Dialect.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/InitAllDialects.h"
#include "aiir/Tools/aiir-lsp-server/AiirLspServerMain.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

int main(int argc, char **argv) {
  aiir::DialectRegistry registry;
  aiir::registerAllDialects(registry);
  registry.insert<cir::CIRDialect>();
  return failed(aiir::AiirLspServerMain(argc, argv, registry));
}
