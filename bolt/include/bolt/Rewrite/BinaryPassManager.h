//===- bolt/Rewrite/BinaryPassManager.h - Binary-level passes ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A very simple binary-level analysis/optimization passes system.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_BINARY_PASS_MANAGER_H
#define BOLT_REWRITE_BINARY_PASS_MANAGER_H

#include "bolt/Passes/BinaryPasses.h"
#include <memory>
#include <vector>

namespace llvm {
namespace bolt {
class BinaryContext;

/// Simple class for managing analyses and optimizations on BinaryFunctions.
class BinaryFunctionPassManager {
private:
  BinaryContext &BC;
  std::vector<std::pair<const bool, std::unique_ptr<BinaryFunctionPass>>>
      Passes;

public:
  static const char TimerGroupName[];
  static const char TimerGroupDesc[];

  BinaryFunctionPassManager(BinaryContext &BC) : BC(BC) {}

  /// Adds a pass to this manager based on the value of its corresponding
  /// command-line option.
  void registerPass(std::unique_ptr<BinaryFunctionPass> Pass, const bool Run) {
    Passes.emplace_back(Run, std::move(Pass));
  }

  /// Adds an unconditionally run pass to this manager.
  void registerPass(std::unique_ptr<BinaryFunctionPass> Pass) {
    Passes.emplace_back(true, std::move(Pass));
  }

  /// Run all registered passes in the order they were added.
  Error runPasses();

  /// Runs all enabled implemented passes on all functions.
  static Error runAllPasses(BinaryContext &BC);
};

} // namespace bolt
} // namespace llvm

#endif // BOLT_REWRITE_BINARY_PASS_MANAGER_H
