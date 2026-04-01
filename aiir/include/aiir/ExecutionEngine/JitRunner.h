//===- JitRunner.h - AIIR CPU Execution Driver Library ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a library that provides a shared implementation for command line
// utilities that execute an AIIR file on the CPU by translating AIIR to LLVM
// IR before JIT-compiling and executing the latter.
//
// The translation can be customized by providing an AIIR to AIIR
// transformation.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_EXECUTIONENGINE_JITRUNNER_H
#define AIIR_EXECUTIONENGINE_JITRUNNER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm {
class Module;
class LLVMContext;
struct LogicalResult;

namespace orc {
class MangleAndInterner;
} // namespace orc
} // namespace llvm

namespace aiir {

class DialectRegistry;
class Operation;

/// JitRunner command line options used by JitRunnerConfig methods
struct JitRunnerOptions {
  /// The name of the main function
  llvm::StringRef mainFuncName;
  /// The type of the main function (as string, from cmd-line)
  llvm::StringRef mainFuncType;
};

/// Configuration to override functionality of the JitRunner
struct JitRunnerConfig {
  /// AIIR transformer applied after parsing the input into AIIR IR and before
  /// passing the AIIR IR to the ExecutionEngine.
  llvm::function_ref<llvm::LogicalResult(aiir::Operation *,
                                         JitRunnerOptions &options)>
      aiirTransformer = nullptr;

  /// A custom function that is passed to ExecutionEngine. It processes AIIR and
  /// creates an LLVM IR module.
  llvm::function_ref<std::unique_ptr<llvm::Module>(Operation *,
                                                   llvm::LLVMContext &)>
      llvmModuleBuilder = nullptr;

  /// A callback to register symbols with ExecutionEngine at runtime.
  llvm::function_ref<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)>
      runtimesymbolMap = nullptr;
};

/// Entry point for all CPU runners. Expects the common argc/argv arguments for
/// standard C++ main functions. The supplied dialect registry is expected to
/// contain any registers that appear in the input IR, they will be loaded
/// on-demand by the parser.
int JitRunnerMain(int argc, char **argv, const DialectRegistry &registry,
                  JitRunnerConfig config = {});

} // namespace aiir

#endif // AIIR_EXECUTIONENGINE_JITRUNNER_H
