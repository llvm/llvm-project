//===- bolt/Passes/ReorderFunctions.h - Reorder functions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_REORDER_FUNCTIONS_H
#define BOLT_PASSES_REORDER_FUNCTIONS_H

#include "bolt/Core/BinaryFunctionCallGraph.h"
#include "bolt/Passes/BinaryPasses.h"
#include "llvm/ADT/DenseSet.h"

namespace llvm {
namespace bolt {
class Cluster;

/// Modify function order for streaming based on hotness.
class ReorderFunctions : public BinaryFunctionPass {
  BinaryFunctionCallGraph Cg;

  void reorder(BinaryContext &BC, std::vector<Cluster> &&Clusters,
               std::map<uint64_t, BinaryFunction> &BFs,
               uint32_t StartIndex = 0);

  /// Read the function order file and assign indices to listed functions.
  /// \p StartIndex is the first index to assign.
  /// \p OrderedFuncs if non-null, will be populated with the set of functions
  ///    that were assigned indices from the order file.
  /// \returns the next available index after all assigned functions, or an
  ///    error if the order file cannot be read.
  Expected<uint32_t> assignFunctionOrder(
      BinaryContext &BC, std::map<uint64_t, BinaryFunction> &BFs,
      uint32_t StartIndex, DenseSet<const BinaryFunction *> *OrderedFuncs);

  void printStats(BinaryContext &BC, const std::vector<Cluster> &Clusters,
                  const std::vector<uint64_t> &FuncAddr);

public:
  enum ReorderType : char {
    RT_NONE = 0,
    RT_EXEC_COUNT,
    RT_HFSORT,
    RT_HFSORT_PLUS,
    RT_CDSORT,
    RT_PETTIS_HANSEN,
    RT_RANDOM,
    RT_USER
  };

  explicit ReorderFunctions(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "reorder-functions"; }
  Error runOnFunctions(BinaryContext &BC) override;

  static Error readFunctionOrderFile(std::vector<std::string> &FunctionNames);
};

} // namespace bolt
} // namespace llvm

#endif // BOLT_PASSES_REORDER_FUNCTIONS_H
