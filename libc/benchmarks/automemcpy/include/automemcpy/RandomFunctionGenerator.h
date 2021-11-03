//===-- Generate random but valid function descriptors  ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_BENCHMARKS_AUTOMEMCPY_RANDOM_FUNCTION_GENERATOR_H
#define LLVM_LIBC_BENCHMARKS_AUTOMEMCPY_RANDOM_FUNCTION_GENERATOR_H

#include "automemcpy/FunctionDescriptor.h"
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/StringRef.h>
#include <vector>
#include <z3++.h>

namespace llvm {
namespace automemcpy {

// Holds the state for the constraint solver.
// It implements a single method that returns the next valid description.
struct RandomFunctionGenerator {
  RandomFunctionGenerator();

  // Get the next valid FunctionDescriptor or llvm::None.
  Optional<FunctionDescriptor> next();

private:
  // Returns an expression where `Variable` is forced to be one of the `Values`.
  z3::expr inSetConstraint(z3::expr &Variable, ArrayRef<int> Values) const;
  // Add constaints to `Begin` and `End` so that they are:
  // - between 0 and kMaxSize (inclusive)
  // - ordered (begin<=End)
  // - amongst a set of predefined values.
  void addBoundsAndAnchors(z3::expr &Begin, z3::expr &End);
  // Add constraints to make sure that the loop block size is amongst a set of
  // predefined values. Also makes sure that the loop that the loop is iterated
  // at least `LoopMinIter` times.
  void addLoopConstraints(const z3::expr &LoopBegin, const z3::expr &LoopEnd,
                          z3::expr &LoopBlockSize, int LoopMinIter);

  z3::context Context;
  z3::solver Solver;

  z3::expr Type;
  z3::expr ContiguousBegin, ContiguousEnd;
  z3::expr OverlapBegin, OverlapEnd;
  z3::expr LoopBegin, LoopEnd, LoopBlockSize;
  z3::expr AlignedLoopBegin, AlignedLoopEnd, AlignedLoopBlockSize,
      AlignedAlignment, AlignedArg;
  z3::expr AcceleratorBegin, AcceleratorEnd;
  z3::expr ElementClass;
};

} // namespace automemcpy
} // namespace llvm

#endif /* LLVM_LIBC_BENCHMARKS_AUTOMEMCPY_RANDOM_FUNCTION_GENERATOR_H */
