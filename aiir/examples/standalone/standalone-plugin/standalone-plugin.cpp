//===- standalone-plugin.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/IR/AIIRContext.h"
#include "aiir/InitAllDialects.h"
#include "aiir/Tools/Plugins/DialectPlugin.h"

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandalonePasses.h"
#include "aiir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

using namespace aiir;

/// Dialect plugin registration mechanism.
/// Observe that it also allows to register passes.
/// Necessary symbol to register the dialect plugin.
extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
aiirGetDialectPluginInfo() {
  return {AIIR_PLUGIN_API_VERSION, "Standalone", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<aiir::standalone::StandaloneDialect>();
            aiir::standalone::registerPasses();
          }};
}

/// Pass plugin registration mechanism.
/// Necessary symbol to register the pass plugin.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo aiirGetPassPluginInfo() {
  return {AIIR_PLUGIN_API_VERSION, "StandalonePasses", LLVM_VERSION_STRING,
          []() { aiir::standalone::registerPasses(); }};
}
