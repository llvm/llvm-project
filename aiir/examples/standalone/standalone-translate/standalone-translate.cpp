//===- standalone-translate.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to AIIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneDialect.h"
#include "aiir/IR/DialectRegistry.h"
#include "aiir/IR/Operation.h"
#include "aiir/InitAllTranslations.h"
#include "aiir/Tools/aiir-translate/AiirTranslateMain.h"
#include "aiir/Tools/aiir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  aiir::registerAllTranslations();

  // TODO: Register standalone translations here.
  aiir::TranslateFromAIIRRegistration withdescription(
      "option", "different from option",
      [](aiir::Operation *op, llvm::raw_ostream &output) {
        return llvm::LogicalResult::success();
      },
      [](aiir::DialectRegistry &a) {});

  return failed(
      aiir::aiirTranslateMain(argc, argv, "AIIR Translation Testing Tool"));
}
