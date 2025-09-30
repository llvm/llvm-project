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
#include "bolt/Core/MCInstUtils.h"
#include "bolt/Passes/BinaryPasses.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace llvm {
namespace bolt {
namespace PAuthGadgetScanner {

// The report classes are designed to be used in an immutable manner.
// When an issue report is constructed in multiple steps, an attempt is made
// to distinguish intermediate and final results at the type level.
//
// Here is an overview of issue life-cycle:
// * an analysis (SrcSafetyAnalysis or DstSafetyAnalysis) computes register
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

/// The set of instructions writing to the affected register in an unsafe
/// manner.
///
/// This is a hint to be printed alongside the report. It should be further
/// analyzed by the user.
class ClobberingInfo : public ExtraInfo {
  SmallVector<MCInstReference> ClobberingInstrs;

public:
  ClobberingInfo(ArrayRef<MCInstReference> Instrs) : ClobberingInstrs(Instrs) {}

  void print(raw_ostream &OS, const MCInstReference Location) const override;
};

/// The set of instructions leaking the authenticated pointer before the
/// result of authentication was checked.
///
/// This is a hint to be printed alongside the report. It should be further
/// analyzed by the user.
class LeakageInfo : public ExtraInfo {
  SmallVector<MCInstReference> LeakingInstrs;

public:
  LeakageInfo(ArrayRef<MCInstReference> Instrs) : LeakingInstrs(Instrs) {}

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

  void findUnsafeDefs(SmallVector<PartialReport<MCPhysReg>> &Reports);
  void augmentUnsafeDefReports(ArrayRef<PartialReport<MCPhysReg>> Reports);

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
