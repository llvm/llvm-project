//===- bolt/Passes/PAuthGadgetScanner.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_PAUTHGADGETSCANNER_H
#define BOLT_PASSES_PAUTHGADGETSCANNER_H

#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Passes/BinaryPasses.h"
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
    return std::tie(BB, BBIndex) < std::tie(RHS.BB, RHS.BBIndex);
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

  static MCInstInBFReference get(const MCInst *Inst, BinaryFunction &BF) {
    for (auto &I : BF.instrs())
      if (Inst == &I.second)
        return MCInstInBFReference(&BF, I.first);
    return {};
  }

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

  static MCInstReference get(const MCInst *Inst, BinaryFunction &BF) {
    if (BF.hasCFG())
      return MCInstInBBReference::get(Inst, BF);
    return MCInstInBFReference::get(Inst, BF);
  }

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

  operator bool() const {
    switch (ParentKind) {
    case BasicBlockParent:
      return U.BBRef.BB != nullptr;
    case FunctionParent:
      return U.BFRef.BF != nullptr;
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

namespace PAuthGadgetScanner {

// The report classes are designed to be used in an immutable manner.
// When an issue report is constructed in multiple steps, an attempt is made
// to distinguish intermediate and final results at the type level.
//
// Here is an overview of issue life-cycle:
// * an analysis (SrcSafetyAnalysis at now, DstSafetyAnalysis will be added
//   later to support the detection of authentication oracles) computes register
//   state for each instruction in the function.
// * for each instruction, it is checked whether it is a gadget of some kind,
//   taking the computed state into account. If a gadget is found, its kind
//   and location are stored into a subclass of Diagnostic wrapped into
//   PartialReport<ReqT>.
// * if any issue is to be reported for the function, the same analysis is
//   re-run to collect extra information to provide to the user. Which extra
//   information can be requested depends on the particular analysis (for
//   example, SrcSafetyAnalysis is able to compute the set of instructions
//   clobbering the particular register, thus ReqT is MCPhysReg). At this stage,
//   `FinalReport`s are created.
//
// Here, the subclasses of Diagnostic store the pieces of information which
// are kept unchanged since they are collected on the first run of the analysis.
// PartialReport<T>::RequestedDetails, on the other hand, is replaced with
// FinalReport::Details computed by the second run of the analysis.

/// Description of a gadget kind that can be detected. Intended to be
/// statically allocated and attached to reports by reference.
class GadgetKind {
  const char *Description;

public:
  /// Wraps a description string which must be a string literal.
  GadgetKind(const char *Description) : Description(Description) {}

  StringRef getDescription() const { return Description; }
};

/// Basic diagnostic information, which is kept unchanged since it is collected
/// on the first run of the analysis.
struct Diagnostic {
  MCInstReference Location;

  Diagnostic(MCInstReference Location) : Location(Location) {}
  virtual ~Diagnostic() {}

  virtual void generateReport(raw_ostream &OS,
                              const BinaryContext &BC) const = 0;

  void printBasicInfo(raw_ostream &OS, const BinaryContext &BC,
                      StringRef IssueKind) const;
};

struct GadgetDiagnostic : public Diagnostic {
  // The particular kind of gadget that is detected.
  const GadgetKind &Kind;

  GadgetDiagnostic(const GadgetKind &Kind, MCInstReference Location)
      : Diagnostic(Location), Kind(Kind) {}

  void generateReport(raw_ostream &OS, const BinaryContext &BC) const override;
};

/// Report with a free-form message attached.
struct GenericDiagnostic : public Diagnostic {
  std::string Text;
  GenericDiagnostic(MCInstReference Location, StringRef Text)
      : Diagnostic(Location), Text(Text) {}
  virtual void generateReport(raw_ostream &OS,
                              const BinaryContext &BC) const override;
};

/// Extra information about an issue collected on the slower, detailed,
/// run of the analysis.
class ExtraInfo {
public:
  virtual void print(raw_ostream &OS, const MCInstReference Location) const = 0;

  virtual ~ExtraInfo() {}
};

class ClobberingInfo : public ExtraInfo {
  SmallVector<MCInstReference> ClobberingInstrs;

public:
  ClobberingInfo(ArrayRef<MCInstReference> Instrs) : ClobberingInstrs(Instrs) {}

  void print(raw_ostream &OS, const MCInstReference Location) const override;
};

/// A brief version of a report that can be further augmented with the details.
///
/// A half-baked report produced on the first run of the analysis. An extra,
/// analysis-specific information may be requested to be collected on the
/// second run.
template <typename T> struct PartialReport {
  PartialReport(std::shared_ptr<Diagnostic> Issue,
                const std::optional<T> RequestedDetails)
      : Issue(Issue), RequestedDetails(RequestedDetails) {}

  std::shared_ptr<Diagnostic> Issue;
  std::optional<T> RequestedDetails;
};

/// A final version of the report.
struct FinalReport {
  FinalReport(std::shared_ptr<Diagnostic> Issue,
              std::shared_ptr<ExtraInfo> Details)
      : Issue(Issue), Details(Details) {}

  std::shared_ptr<Diagnostic> Issue;
  std::shared_ptr<ExtraInfo> Details;
};

struct FunctionAnalysisResult {
  std::vector<FinalReport> Diagnostics;
};

/// A helper class storing per-function context to be instantiated by Analysis.
class FunctionAnalysisContext {
  BinaryContext &BC;
  BinaryFunction &BF;
  MCPlusBuilder::AllocatorIdTy AllocatorId;
  FunctionAnalysisResult Result;

  bool PacRetGadgetsOnly;

  void findUnsafeUses(SmallVector<PartialReport<MCPhysReg>> &Reports);
  void augmentUnsafeUseReports(ArrayRef<PartialReport<MCPhysReg>> Reports);

  /// Process the reports which do not have to be augmented, and remove them
  /// from Reports.
  void handleSimpleReports(SmallVector<PartialReport<MCPhysReg>> &Reports);

public:
  FunctionAnalysisContext(BinaryFunction &BF,
                          MCPlusBuilder::AllocatorIdTy AllocatorId,
                          bool PacRetGadgetsOnly)
      : BC(BF.getBinaryContext()), BF(BF), AllocatorId(AllocatorId),
        PacRetGadgetsOnly(PacRetGadgetsOnly) {}

  void run();

  const FunctionAnalysisResult &getResult() const { return Result; }
};

class Analysis : public BinaryFunctionPass {
  /// Only search for pac-ret violations.
  bool PacRetGadgetsOnly;

  void runOnFunction(BinaryFunction &Function,
                     MCPlusBuilder::AllocatorIdTy AllocatorId);

  std::map<const BinaryFunction *, FunctionAnalysisResult> AnalysisResults;
  std::mutex AnalysisResultsMutex;

public:
  explicit Analysis(bool PacRetGadgetsOnly)
      : BinaryFunctionPass(false), PacRetGadgetsOnly(PacRetGadgetsOnly) {}

  const char *getName() const override { return "pauth-gadget-scanner"; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace PAuthGadgetScanner
} // namespace bolt
} // namespace llvm

#endif
