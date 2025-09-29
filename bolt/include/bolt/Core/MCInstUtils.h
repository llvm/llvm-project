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

} // namespace bolt
} // namespace llvm

#endif
