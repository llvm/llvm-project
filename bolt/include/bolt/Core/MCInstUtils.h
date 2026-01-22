//===- bolt/Core/MCInstUtils.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_MCINSTUTILS_H
#define BOLT_CORE_MCINSTUTILS_H

#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/MCPlus.h"
#include <map>
#include <variant>

namespace llvm {
class MCCodeEmitter;
}

namespace llvm {
namespace bolt {

class BinaryFunction;

/// MCInstReference represents a reference to a constant MCInst as stored either
/// in a BinaryFunction (i.e. before a CFG is created), or in a BinaryBasicBlock
/// (after a CFG is created).
///
/// The reference may be invalidated when the function containing the referenced
/// instruction is modified.
class MCInstReference {
public:
  using nocfg_const_iterator = std::map<uint32_t, MCInst>::const_iterator;

  /// Constructs an empty reference.
  MCInstReference() : Reference(RefInBB(nullptr, /*Index=*/0)) {}

  /// Constructs a reference to the instruction inside the basic block.
  MCInstReference(const BinaryBasicBlock &BB, const MCInst &Inst)
      : Reference(RefInBB(&BB, getInstIndexInBB(BB, Inst))) {}
  /// Constructs a reference to the instruction inside the basic block.
  MCInstReference(const BinaryBasicBlock &BB, unsigned Index)
      : Reference(RefInBB(&BB, Index)) {}

  /// Constructs a reference to the instruction inside the function without
  /// CFG information.
  MCInstReference(const BinaryFunction &BF, nocfg_const_iterator It)
      : Reference(RefInBF(&BF, It)) {}

  /// Locates an instruction inside a function and returns a reference.
  static MCInstReference get(const MCInst &Inst, const BinaryFunction &BF);

  bool operator==(const MCInstReference &Other) const {
    return Reference == Other.Reference;
  }

  const MCInst &getMCInst() const {
    assert(!empty() && "Empty reference");
    if (auto *Ref = tryGetRefInBB()) {
      [[maybe_unused]] unsigned NumInstructions = Ref->BB->size();
      assert(Ref->Index < NumInstructions && "Invalid reference");
      return Ref->BB->getInstructionAtIndex(Ref->Index);
    }
    return getRefInBF().It->second;
  }

  operator const MCInst &() const { return getMCInst(); }

  bool empty() const {
    if (auto *Ref = tryGetRefInBB())
      return Ref->BB == nullptr;
    return getRefInBF().BF == nullptr;
  }

  bool hasCFG() const { return !empty() && tryGetRefInBB() != nullptr; }

  const BinaryFunction *getFunction() const {
    assert(!empty() && "Empty reference");
    if (auto *Ref = tryGetRefInBB())
      return Ref->BB->getFunction();
    return getRefInBF().BF;
  }

  const BinaryBasicBlock *getBasicBlock() const {
    assert(!empty() && "Empty reference");
    if (auto *Ref = tryGetRefInBB())
      return Ref->BB;
    return nullptr;
  }

  /// Computes the original address of the instruction (or offset from base
  /// for PIC), assuming the containing function was not modified.
  ///
  /// This function is intended for the use cases like debug printing, as it
  /// is only as precise as BinaryContext::computeCodeSize() is and requires
  /// iterating over the prefix of the basic block (when CFG is available).
  ///
  /// MCCodeEmitter is not thread safe and the default instance from
  /// BinaryContext is used by default, thus pass an instance explicitly if
  /// this function may be called from multithreaded code.
  uint64_t computeAddress(const MCCodeEmitter *Emitter = nullptr) const;

  raw_ostream &print(raw_ostream &OS) const;

private:
  static unsigned getInstIndexInBB(const BinaryBasicBlock &BB,
                                   const MCInst &Inst) {
    // Usage of pointer arithmetic assumes the instructions are stored in a
    // vector, see BasicBlockStorageIsVector in MCInstUtils.cpp.
    const MCInst *FirstInstInBB = &*BB.begin();
    return &Inst - FirstInstInBB;
  }

  // Two cases are possible:
  // * functions with CFG reconstructed - a function stores a collection of
  //   basic blocks, each basic block stores a contiguous vector of MCInst
  // * functions without CFG - there are no basic blocks created,
  //   the instructions are directly stored in std::map in BinaryFunction
  //
  // In both cases, the direct parent of MCInst is stored together with an
  // index or iterator pointing to the instruction.

  // Helper struct: CFG is available, the direct parent is a basic block.
  struct RefInBB {
    RefInBB(const BinaryBasicBlock *BB, unsigned Index)
        : BB(BB), Index(Index) {}
    RefInBB(const RefInBB &Other) = default;
    RefInBB &operator=(const RefInBB &Other) = default;

    const BinaryBasicBlock *BB;
    unsigned Index;

    bool operator==(const RefInBB &Other) const {
      return BB == Other.BB && Index == Other.Index;
    }
  };

  // Helper struct: CFG is *not* available, the direct parent is a function,
  // iterator's type is std::map<uint32_t, MCInst>::iterator (the mapped value
  // is an instruction's offset).
  struct RefInBF {
    RefInBF(const BinaryFunction *BF, nocfg_const_iterator It)
        : BF(BF), It(It) {}
    RefInBF(const RefInBF &Other) = default;
    RefInBF &operator=(const RefInBF &Other) = default;

    const BinaryFunction *BF;
    nocfg_const_iterator It;

    bool operator==(const RefInBF &Other) const {
      return BF == Other.BF && It->first == Other.It->first;
    }
  };

  std::variant<RefInBB, RefInBF> Reference;

  // Utility methods to be used like this:
  //
  //     if (auto *Ref = tryGetRefInBB())
  //       return Ref->doSomething(...);
  //     return getRefInBF().doSomethingElse(...);
  const RefInBB *tryGetRefInBB() const {
    assert(std::get_if<RefInBB>(&Reference) ||
           std::get_if<RefInBF>(&Reference));
    return std::get_if<RefInBB>(&Reference);
  }
  const RefInBF &getRefInBF() const {
    assert(std::get_if<RefInBF>(&Reference));
    return *std::get_if<RefInBF>(&Reference);
  }
};

static inline raw_ostream &operator<<(raw_ostream &OS,
                                      const MCInstReference &Ref) {
  return Ref.print(OS);
}

/// Instruction-matching helpers operating on a single instruction at a time.
///
/// The idea is to make low-level instruction matching as readable as possible.
/// The classes contained in this namespace are intended to be used as a
/// domain-specific language to match MCInst with the particular opcode and
/// operands.
///
/// The goals of this DSL include
/// * matching a single instruction against the template consisting of the
///   particular target-specific opcode and a pattern of operands
/// * matching operands against the known values (such as 42, AArch64::X1 or
///   "the value of --brk-operand=N command line argument")
/// * capturing operands of an instruction ("whatever is the destination
///   register of AArch64::ADDXri instruction, store it to Xd variable to be
///   queried later")
/// * expressing repeated operands of a single matched instruction (such as
///   "ADDXri Xd, Xd, 42, 0" for an arbitrary register Xd) as well as across
///   multiple calls to matchInst(), which is naturally achieved by sequentially
///   capturing the operands and matching operands against the known values
/// * matching multi-instruction code patterns by sequentially calling
///   matchInst() while passing around already matched operands
///
/// The non-goals (compared to MCPlusBuilder::MCInstMatcher) include
/// * matching an arbitrary tree of instructions in a single matchInst() call
/// * encapsulation of target-specific knowledge ("match an increment of Xm
///   by 42")
///
/// Unlike MCPlusBuilder::MCInstMatcher, this DSL focuses on the use cases when
/// the precise control over the instruction order is important. For example,
/// let's consider a target-specific function that has to match two particular
/// instructions against this pattern (for two different registers Xm and Xn)
///
///     ADDXrs Xm, Xn, Xm, #0
///     BR     Xm
///
/// and return the register holding the branch target. Assuming the instructions
/// are available as MaybeAdd and MaybeBr, the following code can be used:
///
///     // Bring the short names into the local scope:
///     using namespace LowLevelInstMatcherDSL;
///     // Declare the registers to capture:
///     Reg Xn, Xm;
///     // Capture the 0th and 1st operands, match the 2nd operand against the
///     // just captured Xm register, match the 3rd operand against literal 0:
///     if (!matchInst(MaybeAdd, AArch64::ADDXrs, Xm, Xn, Xm, Imm(0))
///       return AArch64::NoRegister;
///     // Match the 0th operand against Xm:
///     if (!matchInst(MaybeBr, AArch64::BR, Xm))
///       return AArch64::NoRegister;
///     // Manually check that Xm and Xn did not match the same register:
///     if (Xm.get() == Xn.get())
///       return AArch64::NoRegister;
///     // Return the matched register:
///     return Xm.get();
///
namespace LowLevelInstMatcherDSL {

// The base class to match an operand of type T.
//
// The subclasses of OpMatcher are intended to be allocated on the stack and
// to only be used by passing them to matchInst() and by calling their get()
// function, thus the peculiar `mutable` specifiers: to make the calling code
// compact and readable, the templated matchInst() function has to accept both
// long-lived Imm/Reg wrappers declared as local variables (intended to capture
// the first operand's value and match the subsequent operands, whether inside
// a single instruction or across multiple instructions), as well as temporary
// wrappers around literal values to match, f.e. Imm(42) or Reg(AArch64::XZR).
template <typename T> class OpMatcher {
  mutable std::optional<T> Value;
  mutable std::optional<T> SavedValue;

  // Remember/restore the last Value - to be called by matchInst.
  void remember() const { SavedValue = Value; }
  void restore() const { Value = SavedValue; }

  template <class... OpMatchers>
  friend bool matchInst(const MCInst &, unsigned, const OpMatchers &...);

protected:
  OpMatcher(std::optional<T> ValueToMatch) : Value(ValueToMatch) {}

  bool matchValue(T OpValue) const {
    // Check that OpValue does not contradict the existing Value.
    bool MatchResult = !Value || *Value == OpValue;
    // If MatchResult is false, all matchers will be reset before returning from
    // matchInst, including this one, thus no need to assign conditionally.
    Value = OpValue;

    return MatchResult;
  }

public:
  /// Returns the captured value.
  T get() const {
    assert(Value.has_value());
    return *Value;
  }
};

class Reg : public OpMatcher<MCPhysReg> {
  bool matches(const MCOperand &Op) const {
    if (!Op.isReg())
      return false;

    return matchValue(Op.getReg());
  }

  template <class... OpMatchers>
  friend bool matchInst(const MCInst &, unsigned, const OpMatchers &...);

public:
  Reg(std::optional<MCPhysReg> RegToMatch = std::nullopt)
      : OpMatcher<MCPhysReg>(RegToMatch) {}
};

class Imm : public OpMatcher<int64_t> {
  bool matches(const MCOperand &Op) const {
    if (!Op.isImm())
      return false;

    return matchValue(Op.getImm());
  }

  template <class... OpMatchers>
  friend bool matchInst(const MCInst &, unsigned, const OpMatchers &...);

public:
  Imm(std::optional<int64_t> ImmToMatch = std::nullopt)
      : OpMatcher<int64_t>(ImmToMatch) {}
};

/// Tries to match Inst and updates Ops on success.
///
/// If Inst has the specified Opcode and its operand list prefix matches Ops,
/// this function returns true and updates Ops, otherwise false is returned and
/// values of Ops are kept as before matchInst was called.
///
/// Please note that while Ops are technically passed by a const reference to
/// make invocations like `matchInst(MI, Opcode, Imm(42))` possible, all their
/// fields are marked mutable.
template <class... OpMatchers>
bool matchInst(const MCInst &Inst, unsigned Opcode, const OpMatchers &...Ops) {
  if (Inst.getOpcode() != Opcode)
    return false;
  assert(sizeof...(Ops) <= MCPlus::getNumPrimeOperands(Inst) &&
         "Too many operands are matched for the Opcode");

  // Ask each matcher to remember its current value in case of rollback.
  (Ops.remember(), ...);

  // Check if all matchers match the corresponding operands.
  auto It = Inst.begin();
  auto AllMatched = (Ops.matches(*(It++)) && ... && true);

  // If match failed, restore the original captured values.
  if (!AllMatched) {
    (Ops.restore(), ...);
    return false;
  }

  return true;
}

} // namespace LowLevelInstMatcherDSL

} // namespace bolt
} // namespace llvm

#endif
