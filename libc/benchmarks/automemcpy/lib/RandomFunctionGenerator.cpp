//===-- Generate random but valid function descriptors  -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "automemcpy/RandomFunctionGenerator.h"

#include <llvm/ADT/None.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include <set>

namespace llvm {
namespace automemcpy {

// Exploration parameters
// ----------------------
// Here we define a set of values that will contraint the exploration and
// limit combinatorial explosion.

// We limit the number of cases for individual sizes to sizes up to 4.
// More individual sizes don't bring much over the overlapping strategy.
static constexpr int kMaxIndividualSize = 4;

// We limit Overlapping Strategy to sizes up to 256.
// An overlap of 256B means accessing 128B at once which is usually not
// feasible by current CPUs. We rely on the compiler to generate multiple
// loads/stores if needed but higher sizes are unlikely to benefit from hardware
// acceleration.
static constexpr int kMaxOverlapSize = 256;

// For the loop strategies, we make sure that they iterate at least a certain
// number of times to amortize the cost of looping.
static constexpr int kLoopMinIter = 3;
static constexpr int kAlignedLoopMinIter = 2;

// We restrict the size of the block of data to handle in a loop.
// Generally speaking block size <= 16 perform poorly.
static constexpr int kLoopBlockSize[] = {16, 32, 64};

// We restrict alignment to the following values.
static constexpr int kLoopAlignments[] = {16, 32, 64};

// We make sure that the region bounds are one of the following values.
static constexpr int kAnchors[] = {0,  1,  2,   4,   8,   16,   32,      48,
                                   64, 96, 128, 256, 512, 1024, kMaxSize};

// We also allow disabling loops, aligned loops and accelerators.
static constexpr bool kDisableLoop = false;
static constexpr bool kDisableAlignedLoop = false;
static constexpr bool kDisableAccelerator = false;

// For memcpy, we can also explore whether aligning on source or destination has
// an effect.
static constexpr bool kExploreAlignmentArg = true;

// The function we generate code for.
// BCMP is specifically disabled for now.
static constexpr int kFunctionTypes[] = {
    (int)FunctionType::MEMCPY,
    (int)FunctionType::MEMCMP,
    //  (int)FunctionType::BCMP,
    (int)FunctionType::MEMSET,
    (int)FunctionType::BZERO,
};

// The actual implementation of each function can be handled via primitive types
// (SCALAR), vector types where available (NATIVE) or by the compiler (BUILTIN).
// We want to move toward delegating the code generation entirely to the
// compiler but for now we have to make use of -per microarchitecture- custom
// implementations. Scalar being more portable but also less performant, we
// remove it as well.
static constexpr int kElementClasses[] = {
    // (int)ElementTypeClass::SCALAR,
    (int)ElementTypeClass::NATIVE,
    // (int)ElementTypeClass::BUILTIN
};

RandomFunctionGenerator::RandomFunctionGenerator()
    : Solver(Context), Type(Context.int_const("Type")),
      ContiguousBegin(Context.int_const("ContiguousBegin")),
      ContiguousEnd(Context.int_const("ContiguousEnd")),
      OverlapBegin(Context.int_const("OverlapBegin")),
      OverlapEnd(Context.int_const("OverlapEnd")),
      LoopBegin(Context.int_const("LoopBegin")),
      LoopEnd(Context.int_const("LoopEnd")),
      LoopBlockSize(Context.int_const("LoopBlockSize")),
      AlignedLoopBegin(Context.int_const("AlignedLoopBegin")),
      AlignedLoopEnd(Context.int_const("AlignedLoopEnd")),
      AlignedLoopBlockSize(Context.int_const("AlignedLoopBlockSize")),
      AlignedAlignment(Context.int_const("AlignedAlignment")),
      AlignedArg(Context.int_const("AlignedArg")),
      AcceleratorBegin(Context.int_const("AcceleratorBegin")),
      AcceleratorEnd(Context.int_const("AcceleratorEnd")),
      ElementClass(Context.int_const("ElementClass")) {
  // All possible functions.
  Solver.add(inSetConstraint(Type, kFunctionTypes));

  // Add constraints for region bounds.
  addBoundsAndAnchors(ContiguousBegin, ContiguousEnd);
  addBoundsAndAnchors(OverlapBegin, OverlapEnd);
  addBoundsAndAnchors(LoopBegin, LoopEnd);
  addBoundsAndAnchors(AlignedLoopBegin, AlignedLoopEnd);
  addBoundsAndAnchors(AcceleratorBegin, AcceleratorEnd);
  // We always consider strategies in this order, and we
  // always end with the `Accelerator` strategy, as it's typically more
  // efficient for large sizes.
  // Contiguous <= Overlap <= Loop <= AlignedLoop <= Accelerator
  Solver.add(ContiguousEnd == OverlapBegin);
  Solver.add(OverlapEnd == LoopBegin);
  Solver.add(LoopEnd == AlignedLoopBegin);
  Solver.add(AlignedLoopEnd == AcceleratorBegin);
  // Fix endpoints: The minimum size that we want to copy is 0, and we always
  // start with the `Contiguous` strategy. The max size is `kMaxSize`.
  Solver.add(ContiguousBegin == 0);
  Solver.add(AcceleratorEnd == kMaxSize);
  // Contiguous
  Solver.add(ContiguousEnd <= kMaxIndividualSize + 1);
  // Overlap
  Solver.add(OverlapEnd <= kMaxOverlapSize + 1);
  // Overlap only ever makes sense when accessing multiple bytes at a time.
  // i.e. Overlap<1> is useless.
  Solver.add(OverlapBegin == OverlapEnd || OverlapBegin >= 2);
  // Loop
  addLoopConstraints(LoopBegin, LoopEnd, LoopBlockSize, kLoopMinIter);
  // Aligned Loop
  addLoopConstraints(AlignedLoopBegin, AlignedLoopEnd, AlignedLoopBlockSize,
                     kAlignedLoopMinIter);
  Solver.add(inSetConstraint(AlignedAlignment, kLoopAlignments));
  Solver.add(AlignedLoopBegin == AlignedLoopEnd || AlignedLoopBegin >= 64);
  Solver.add(AlignedLoopBlockSize >= AlignedAlignment);
  Solver.add(AlignedLoopBlockSize >= LoopBlockSize);
  z3::expr IsMemcpy = Type == (int)FunctionType::MEMCPY;
  z3::expr ExploreAlignment = IsMemcpy && kExploreAlignmentArg;
  Solver.add(
      (ExploreAlignment &&
       inSetConstraint(AlignedArg, {(int)AlignArg::_1, (int)AlignArg::_2})) ||
      (!ExploreAlignment && AlignedArg == (int)AlignArg::_1));
  // Accelerator
  Solver.add(IsMemcpy ||
             (AcceleratorBegin ==
              AcceleratorEnd)); // Only Memcpy has accelerator for now.
  // Element classes
  Solver.add(inSetConstraint(ElementClass, kElementClasses));

  if (kDisableLoop)
    Solver.add(LoopBegin == LoopEnd);
  if (kDisableAlignedLoop)
    Solver.add(AlignedLoopBegin == AlignedLoopEnd);
  if (kDisableAccelerator)
    Solver.add(AcceleratorBegin == AcceleratorEnd);
}

// Creates SizeSpan from Begin/End values.
// Returns llvm::None if Begin==End.
static Optional<SizeSpan> AsSizeSpan(size_t Begin, size_t End) {
  if (Begin == End)
    return None;
  SizeSpan SS;
  SS.Begin = Begin;
  SS.End = End;
  return SS;
}

// Generic method to create a `Region` struct with a Span or None if span is
// empty.
template <typename Region>
static Optional<Region> As(size_t Begin, size_t End) {
  if (auto Span = AsSizeSpan(Begin, End)) {
    Region Output;
    Output.Span = *Span;
    return Output;
  }
  return None;
}

// Returns a Loop struct or None if span is empty.
static Optional<Loop> AsLoop(size_t Begin, size_t End, size_t BlockSize) {
  if (auto Span = AsSizeSpan(Begin, End)) {
    Loop Output;
    Output.Span = *Span;
    Output.BlockSize = BlockSize;
    return Output;
  }
  return None;
}

// Returns an AlignedLoop struct or None if span is empty.
static Optional<AlignedLoop> AsAlignedLoop(size_t Begin, size_t End,
                                           size_t BlockSize, size_t Alignment,
                                           AlignArg AlignTo) {
  if (auto Loop = AsLoop(Begin, End, BlockSize)) {
    AlignedLoop Output;
    Output.Loop = *Loop;
    Output.Alignment = Alignment;
    Output.AlignTo = AlignTo;
    return Output;
  }
  return None;
}

Optional<FunctionDescriptor> RandomFunctionGenerator::next() {
  if (Solver.check() != z3::sat)
    return {};

  z3::model m = Solver.get_model();

  // Helper method to get the current numerical value of a z3::expr.
  const auto E = [&m](z3::expr &V) -> int {
    return m.eval(V).get_numeral_int();
  };

  // Fill is the function descriptor to return.
  FunctionDescriptor R;
  R.Type = FunctionType(E(Type));
  R.Contiguous = As<Contiguous>(E(ContiguousBegin), E(ContiguousEnd));
  R.Overlap = As<Overlap>(E(OverlapBegin), E(OverlapEnd));
  R.Loop = AsLoop(E(LoopBegin), E(LoopEnd), E(LoopBlockSize));
  R.AlignedLoop = AsAlignedLoop(E(AlignedLoopBegin), E(AlignedLoopEnd),
                                E(AlignedLoopBlockSize), E(AlignedAlignment),
                                AlignArg(E(AlignedArg)));
  R.Accelerator = As<Accelerator>(E(AcceleratorBegin), E(AcceleratorEnd));
  R.ElementClass = ElementTypeClass(E(ElementClass));

  // Express current state as a set of constraints.
  z3::expr CurrentLayout =
      (Type == E(Type)) && (ContiguousBegin == E(ContiguousBegin)) &&
      (ContiguousEnd == E(ContiguousEnd)) &&
      (OverlapBegin == E(OverlapBegin)) && (OverlapEnd == E(OverlapEnd)) &&
      (LoopBegin == E(LoopBegin)) && (LoopEnd == E(LoopEnd)) &&
      (LoopBlockSize == E(LoopBlockSize)) &&
      (AlignedLoopBegin == E(AlignedLoopBegin)) &&
      (AlignedLoopEnd == E(AlignedLoopEnd)) &&
      (AlignedLoopBlockSize == E(AlignedLoopBlockSize)) &&
      (AlignedAlignment == E(AlignedAlignment)) &&
      (AlignedArg == E(AlignedArg)) &&
      (AcceleratorBegin == E(AcceleratorBegin)) &&
      (AcceleratorEnd == E(AcceleratorEnd)) &&
      (ElementClass == E(ElementClass));

  // Ask solver to never show this configuration ever again.
  Solver.add(!CurrentLayout);
  return R;
}

// Make sure `Variable` is one of the provided values.
z3::expr RandomFunctionGenerator::inSetConstraint(z3::expr &Variable,
                                                  ArrayRef<int> Values) const {
  z3::expr_vector Args(Variable.ctx());
  for (int Value : Values)
    Args.push_back(Variable == Value);
  return z3::mk_or(Args);
}

void RandomFunctionGenerator::addBoundsAndAnchors(z3::expr &Begin,
                                                  z3::expr &End) {
  // Begin and End are picked amongst a set of predefined values.
  Solver.add(inSetConstraint(Begin, kAnchors));
  Solver.add(inSetConstraint(End, kAnchors));
  Solver.add(Begin >= 0);
  Solver.add(Begin <= End);
  Solver.add(End <= kMaxSize);
}

void RandomFunctionGenerator::addLoopConstraints(const z3::expr &LoopBegin,
                                                 const z3::expr &LoopEnd,
                                                 z3::expr &LoopBlockSize,
                                                 int LoopMinIter) {
  Solver.add(inSetConstraint(LoopBlockSize, kLoopBlockSize));
  Solver.add(LoopBegin == LoopEnd ||
             (LoopBegin > (LoopMinIter * LoopBlockSize)));
}

} // namespace automemcpy
} // namespace llvm
