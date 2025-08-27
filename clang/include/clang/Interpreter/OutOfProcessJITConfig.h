//===-- OutOfProcessJITConfig.h - Struct for Out-Of-Process JIT--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a struct that holds configuration options for
// out-of-process JIT execution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_OUTOFPROCESSJITCONFIG_H
#define LLVM_CLANG_INTERPRETER_OUTOFPROCESSJITCONFIG_H

#include <functional>
#include <string>
#include <utility>

namespace clang {

/// \brief Configuration options for out-of-process JIT execution.
struct OutOfProcessJITConfig {
  /// Indicates whether out-of-process JIT execution is enabled.
  bool IsOutOfProcess = false;

  /// Path to the out-of-process JIT executor.
  std::string OOPExecutor;

  std::string OOPExecutorConnect;

  /// Indicates whether to use shared memory for communication.
  bool UseSharedMemory;

  /// String representing the slab allocation size for memory management.
  std::string SlabAllocateSizeString;

  /// Path to the ORC runtime library.
  std::string OrcRuntimePath;
};

} // namespace clang

#endif // LLVM_CLANG_INTERPRETER_OUTOFPROCESSJITCONFIG_H