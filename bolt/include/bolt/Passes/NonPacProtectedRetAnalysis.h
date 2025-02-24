//===- bolt/Passes/NonPacProtectedRetAnalysis.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_NONPACPROTECTEDRETANALYSIS_H
#define BOLT_PASSES_NONPACPROTECTEDRETANALYSIS_H

#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Passes/BinaryPasses.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace llvm {
namespace bolt {

/// @brief  MCInstReference represents a reference to an MCInst as stored either
/// in a BinaryFunction (i.e. before a CFG is created), or in a BinaryBasicBlock
/// (after a CFG is created). It aims to store the necessary information to be
/// able to find the specific MCInst in either the BinaryFunction or
/// BinaryBasicBlock data structures later, so that e.g. the InputAddress of
/// the corresponding instruction can be computed.

struct MCInstInBBReference {
  BinaryBasicBlock *BB;
  int64_t BBIndex;
  MCInstInBBReference(BinaryBasicBlock *BB, int64_t BBIndex)
      : BB(BB), BBIndex(BBIndex) {}
  MCInstInBBReference() : BB(nullptr), BBIndex(0) {}
  static MCInstInBBReference get(const MCInst *Inst, BinaryFunction &BF) {
    for (BinaryBasicBlock &BB : BF)
      for (size_t I = 0; I < BB.size(); ++I)
        if (Inst == &BB.getInstructionAtIndex(I))
          return MCInstInBBReference(&BB, I);
    return {};
  }
  bool operator==(const MCInstInBBReference &RHS) const {
    return BB == RHS.BB && BBIndex == RHS.BBIndex;
  }
  bool operator<(const MCInstInBBReference &RHS) const {
    if (BB != RHS.BB)
      return BB < RHS.BB;
    return BBIndex < RHS.BBIndex;
  }
  operator MCInst &() const {
    assert(BB != nullptr);
    return BB->getInstructionAtIndex(BBIndex);
  }
  uint64_t getAddress() const {
    // 4 bytes per instruction on AArch64.
    // FIXME: the assumption of 4 byte per instruction needs to be fixed before
    // this method gets used on any non-AArch64 binaries (but should be fine for
    // pac-ret analysis, as that is an AArch64-specific feature).
    return BB->getFunction()->getAddress() + BB->getOffset() + BBIndex * 4;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const MCInstInBBReference &);

struct MCInstInBFReference {
  BinaryFunction *BF;
  uint64_t Offset;
  MCInstInBFReference(BinaryFunction *BF, uint64_t Offset)
      : BF(BF), Offset(Offset) {}
  MCInstInBFReference() : BF(nullptr), Offset(0) {}
  bool operator==(const MCInstInBFReference &RHS) const {
    return BF == RHS.BF && Offset == RHS.Offset;
  }
  bool operator<(const MCInstInBFReference &RHS) const {
    if (BF != RHS.BF)
      return BF < RHS.BF;
    return Offset < RHS.Offset;
  }
  operator MCInst &() const {
    assert(BF != nullptr);
    return *BF->getInstructionAtOffset(Offset);
  }

  uint64_t getOffset() const { return Offset; }

  uint64_t getAddress() const { return BF->getAddress() + getOffset(); }
};

raw_ostream &operator<<(raw_ostream &OS, const MCInstInBFReference &);

struct MCInstReference {
  enum Kind { FunctionParent, BasicBlockParent };
  Kind ParentKind;
  union U {
    MCInstInBBReference BBRef;
    MCInstInBFReference BFRef;
    U(MCInstInBBReference BBRef) : BBRef(BBRef) {}
    U(MCInstInBFReference BFRef) : BFRef(BFRef) {}
  } U;
  MCInstReference(MCInstInBBReference BBRef)
      : ParentKind(BasicBlockParent), U(BBRef) {}
  MCInstReference(MCInstInBFReference BFRef)
      : ParentKind(FunctionParent), U(BFRef) {}
  MCInstReference(BinaryBasicBlock *BB, int64_t BBIndex)
      : MCInstReference(MCInstInBBReference(BB, BBIndex)) {}
  MCInstReference(BinaryFunction *BF, uint32_t Offset)
      : MCInstReference(MCInstInBFReference(BF, Offset)) {}

  bool operator<(const MCInstReference &RHS) const {
    if (ParentKind != RHS.ParentKind)
      return ParentKind < RHS.ParentKind;
    switch (ParentKind) {
    case BasicBlockParent:
      return U.BBRef < RHS.U.BBRef;
    case FunctionParent:
      return U.BFRef < RHS.U.BFRef;
    }
    llvm_unreachable("");
  }

  bool operator==(const MCInstReference &RHS) const {
    if (ParentKind != RHS.ParentKind)
      return false;
    switch (ParentKind) {
    case BasicBlockParent:
      return U.BBRef == RHS.U.BBRef;
    case FunctionParent:
      return U.BFRef == RHS.U.BFRef;
    }
    llvm_unreachable("");
  }

  operator MCInst &() const {
    switch (ParentKind) {
    case BasicBlockParent:
      return U.BBRef;
    case FunctionParent:
      return U.BFRef;
    }
    llvm_unreachable("");
  }

  uint64_t getAddress() const {
    switch (ParentKind) {
    case BasicBlockParent:
      return U.BBRef.getAddress();
    case FunctionParent:
      return U.BFRef.getAddress();
    }
    llvm_unreachable("");
  }

  BinaryFunction *getFunction() const {
    switch (ParentKind) {
    case FunctionParent:
      return U.BFRef.BF;
    case BasicBlockParent:
      return U.BBRef.BB->getFunction();
    }
    llvm_unreachable("");
  }

  BinaryBasicBlock *getBasicBlock() const {
    switch (ParentKind) {
    case FunctionParent:
      return nullptr;
    case BasicBlockParent:
      return U.BBRef.BB;
    }
    llvm_unreachable("");
  }
};

raw_ostream &operator<<(raw_ostream &OS, const MCInstReference &);

struct GeneralDiagnostic {
  std::string Text;
  GeneralDiagnostic(const std::string &Text) : Text(Text) {}
  bool operator==(const GeneralDiagnostic &RHS) const {
    return Text == RHS.Text;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const GeneralDiagnostic &Diag);

namespace NonPacProtectedRetAnalysis {
struct Annotation {
  MCInstReference RetInst;
  Annotation(MCInstReference RetInst) : RetInst(RetInst) {}
  virtual bool operator==(const Annotation &RHS) const {
    return RetInst == RHS.RetInst;
  }
  Annotation &operator=(const Annotation &Other) {
    if (this == &Other)
      return *this;
    RetInst = Other.RetInst;
    return *this;
  }
  virtual ~Annotation() {}
  virtual void generateReport(raw_ostream &OS,
                              const BinaryContext &BC) const = 0;
};

struct Gadget : public Annotation {
  std::vector<MCInstReference> OverwritingRetRegInst;
  virtual bool operator==(const Gadget &RHS) const {
    return Annotation::operator==(RHS) &&
           OverwritingRetRegInst == RHS.OverwritingRetRegInst;
  }
  Gadget(MCInstReference RetInst,
         const std::vector<MCInstReference> &OverwritingRetRegInst)
      : Annotation(RetInst), OverwritingRetRegInst(OverwritingRetRegInst) {}
  virtual void generateReport(raw_ostream &OS,
                              const BinaryContext &BC) const override;
};

struct GenDiag : public Annotation {
  GeneralDiagnostic Diag;
  virtual bool operator==(const GenDiag &RHS) const {
    return Annotation::operator==(RHS) && Diag == RHS.Diag;
  }
  GenDiag(MCInstReference RetInst, const std::string &Text)
      : Annotation(RetInst), Diag(Text) {}
  virtual void generateReport(raw_ostream &OS,
                              const BinaryContext &BC) const override;
};

class PacRetAnalysis;

struct FunctionAnalysisResult {
  SmallSet<MCPhysReg, 1> RegistersAffected;
  std::vector<std::shared_ptr<Annotation>> Diagnostics;
};

class Analysis : public BinaryFunctionPass {
  void runOnFunction(BinaryFunction &Function,
                     MCPlusBuilder::AllocatorIdTy AllocatorId);
  FunctionAnalysisResult
  computeDfState(PacRetAnalysis &PRA, BinaryFunction &BF,
                 MCPlusBuilder::AllocatorIdTy AllocatorId);

  std::map<const BinaryFunction *, FunctionAnalysisResult> AnalysisResults;
  std::mutex AnalysisResultsMutex;

public:
  explicit Analysis() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "non-pac-protected-rets"; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace NonPacProtectedRetAnalysis
} // namespace bolt
} // namespace llvm

#endif
