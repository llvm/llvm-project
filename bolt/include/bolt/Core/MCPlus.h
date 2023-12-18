//===- bolt/Core/MCPlus.h - Helpers for MCPlus instructions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations for helper functions for adding annotations
// to MCInst objects.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_MCPLUS_H
#define BOLT_CORE_MCPLUS_H

#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Casting.h"
#include <vector>

namespace llvm {
namespace bolt {

// NOTE: using SmallVector for instruction list results in a memory regression.
using InstructionListType = std::vector<MCInst>;

namespace MCPlus {

/// This type represents C++ EH info for a callsite.  The symbol is the landing
/// pad and the uint64_t represents the action.
using MCLandingPad = std::pair<const MCSymbol *, uint64_t>;

/// An extension to MCInst is provided via extra operands, i.e. operands that
/// are not used in the instruction assembly. Any kind of metadata can be
/// attached to MCInst with this "annotation" extension using MCPlusBuilder
/// interface.
//
/// The first extra operand must be of type kInst with an empty (nullptr)
/// value. The kInst operand type is unused on most non-VLIW architectures.
/// We use it to mark the beginning of annotations operands. The rest of the
/// operands are of Immediate type with annotation info encoded into the value
/// of the immediate.
///
/// There are 2 distinct groups of annotations. The first group is a first-class
/// annotation that affects semantics of the instruction, such as an
/// exception-handling or jump table information. The second group contains
/// information that is supplement, and could be discarded without affecting
/// correctness of the program. Debugging information, and profile information
/// belong to the second group.
///
/// Note: some optimization/transformation passes could use generic annotations
///       inside the pass and remove these annotations after the pass. In this
///       case, the internal state saved with annotations could affect the
///       correctness.
///
/// For the first group, we use a reserved annotation index. Operands in
/// the first groups store a value of an annotation in the immediate field
/// of their corresponding operand.
///
/// Annotations in the second group could be addressed either by name, or by
/// by index which could be queried by providing the name.
class MCAnnotation {
public:
  enum Kind {
    kEHLandingPad,        /// Exception handling landing pad.
    kEHAction,            /// Action for exception handler.
    kGnuArgsSize,         /// GNU args size.
    kJumpTable,           /// Jump Table.
    kTailCall,            /// Tail call.
    kConditionalTailCall, /// CTC.
    kOffset,              /// Offset in the function.
    kLabel,               /// MCSymbol pointing to this instruction.
    kSize,                /// Size of the instruction.
    kGeneric              /// First generic annotation.
  };

  virtual void print(raw_ostream &OS) const = 0;
  virtual bool equals(const MCAnnotation &) const = 0;
  virtual ~MCAnnotation() {}

protected:
  MCAnnotation() {}

private:
  // noncopyable
  MCAnnotation(const MCAnnotation &Other) = delete;
  MCAnnotation &operator=(const MCAnnotation &Other) = delete;
};

/// Instances of this class represent a simple annotation with a
/// specific value type.
/// Note that if ValueType contains any heap allocated memory, it will
/// only be freed if the annotation is removed with the
/// MCPlusBuilder::removeAnnotation method.  This is because all
/// annotations are arena allocated.
template <typename ValueType> class MCSimpleAnnotation : public MCAnnotation {
public:
  ValueType &getValue() { return Value; }
  bool equals(const MCAnnotation &Other) const override {
    return Value == static_cast<const MCSimpleAnnotation &>(Other).Value;
  }
  explicit MCSimpleAnnotation(const ValueType &Val) : Value(Val) {}

  void print(raw_ostream &OS) const override { OS << Value; }

private:
  ValueType Value;
};

/// Return a number of operands in \Inst excluding operands representing
/// annotations.
inline unsigned getNumPrimeOperands(const MCInst &Inst) {
  for (signed I = Inst.getNumOperands() - 1; I >= 0; --I) {
    if (Inst.getOperand(I).isInst())
      return I;
    if (!Inst.getOperand(I).isImm())
      return Inst.getNumOperands();
  }
  return Inst.getNumOperands();
}

/// Return iterator range of operands excluding operands representing
/// annotations.
inline iterator_range<MCInst::iterator> primeOperands(MCInst &Inst) {
  return iterator_range<MCInst::iterator>(
      Inst.begin(), Inst.begin() + getNumPrimeOperands(Inst));
}

inline iterator_range<MCInst::const_iterator>
primeOperands(const MCInst &Inst) {
  return iterator_range<MCInst::const_iterator>(
      Inst.begin(), Inst.begin() + getNumPrimeOperands(Inst));
}

} // namespace MCPlus

} // namespace bolt
} // namespace llvm

#endif
