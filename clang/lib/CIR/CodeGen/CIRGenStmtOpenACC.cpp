//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenACC Stmt nodes as CIR code.
//
//===----------------------------------------------------------------------===//
#include <type_traits>

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "clang/AST/OpenACCClause.h"
#include "clang/AST/StmtOpenACC.h"

#include "mlir/Dialect/OpenACC/OpenACC.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;
using namespace mlir::acc;

namespace {
// Simple type-trait to see if the first template arg is one of the list, so we
// can tell whether to `if-constexpr` a bunch of stuff.
template <typename ToTest, typename T, typename... Tys>
constexpr bool isOneOfTypes =
    std::is_same_v<ToTest, T> || isOneOfTypes<ToTest, Tys...>;
template <typename ToTest, typename T>
constexpr bool isOneOfTypes<ToTest, T> = std::is_same_v<ToTest, T>;

class OpenACCClauseCIREmitter final
    : public OpenACCClauseVisitor<OpenACCClauseCIREmitter> {
  CIRGenModule &cgm;
  // This is necessary since a few of the clauses emit differently based on the
  // directive kind they are attached to.
  OpenACCDirectiveKind dirKind;
  SourceLocation dirLoc;

  struct AttributeData {
    // Value of the 'default' attribute, added on 'data' and 'compute'/etc
    // constructs as a 'default-attr'.
    std::optional<ClauseDefaultValue> defaultVal = std::nullopt;
    // For directives that have their device type architectures listed in
    // attributes (init/shutdown/etc), the list of architectures to be emitted.
    llvm::SmallVector<mlir::acc::DeviceType> deviceTypeArchs{};
  } attrData;

  void clauseNotImplemented(const OpenACCClause &c) {
    cgm.errorNYI(c.getSourceRange(), "OpenACC Clause", c.getClauseKind());
  }

public:
  OpenACCClauseCIREmitter(CIRGenModule &cgm, OpenACCDirectiveKind dirKind,
                          SourceLocation dirLoc)
      : cgm(cgm), dirKind(dirKind), dirLoc(dirLoc) {}

  void VisitClause(const OpenACCClause &clause) {
    clauseNotImplemented(clause);
  }

  void VisitDefaultClause(const OpenACCDefaultClause &clause) {
    switch (clause.getDefaultClauseKind()) {
    case OpenACCDefaultClauseKind::None:
      attrData.defaultVal = ClauseDefaultValue::None;
      break;
    case OpenACCDefaultClauseKind::Present:
      attrData.defaultVal = ClauseDefaultValue::Present;
      break;
    case OpenACCDefaultClauseKind::Invalid:
      break;
    }
  }

  mlir::acc::DeviceType decodeDeviceType(const IdentifierInfo *ii) {
    // '*' case leaves no identifier-info, just a nullptr.
    if (!ii)
      return mlir::acc::DeviceType::Star;
    return llvm::StringSwitch<mlir::acc::DeviceType>(ii->getName())
        .CaseLower("default", mlir::acc::DeviceType::Default)
        .CaseLower("host", mlir::acc::DeviceType::Host)
        .CaseLower("multicore", mlir::acc::DeviceType::Multicore)
        .CasesLower("nvidia", "acc_device_nvidia",
                    mlir::acc::DeviceType::Nvidia)
        .CaseLower("radeon", mlir::acc::DeviceType::Radeon);
  }

  void VisitDeviceTypeClause(const OpenACCDeviceTypeClause &clause) {

    switch (dirKind) {
    case OpenACCDirectiveKind::Init:
    case OpenACCDirectiveKind::Set:
    case OpenACCDirectiveKind::Shutdown: {
      // Device type has a list that is either a 'star' (emitted as 'star'),
      // or an identifer list, all of which get added for attributes.

      for (const DeviceTypeArgument &arg : clause.getArchitectures())
        attrData.deviceTypeArchs.push_back(decodeDeviceType(arg.first));
      break;
    }
    default:
      return clauseNotImplemented(clause);
    }
  }

  // Apply any of the clauses that resulted in an 'attribute'.
  template <typename Op>
  void applyAttributes(CIRGenBuilderTy &builder, Op &op) {

    if (attrData.defaultVal.has_value()) {
      // FIXME: OpenACC: as we implement this for other directive kinds, we have
      // to expand this list.
      // This type-trait checks if 'op'(the first arg) is one of the mlir::acc
      // operations listed in the rest of the arguments.
      if constexpr (isOneOfTypes<Op, ParallelOp, SerialOp, KernelsOp, DataOp>)
        op.setDefaultAttr(*attrData.defaultVal);
      else
        cgm.errorNYI(dirLoc, "OpenACC 'default' clause lowering for ", dirKind);
    }

    if (!attrData.deviceTypeArchs.empty()) {
      // FIXME: OpenACC: as we implement this for other directive kinds, we have
      // to expand this list, or more likely, have a 'noop' branch as most other
      // uses of this apply to the operands instead.
      // This type-trait checks if 'op'(the first arg) is one of the mlir::acc
      if constexpr (isOneOfTypes<Op, InitOp, ShutdownOp>) {
        llvm::SmallVector<mlir::Attribute> deviceTypes;
        for (mlir::acc::DeviceType DT : attrData.deviceTypeArchs)
          deviceTypes.push_back(
              mlir::acc::DeviceTypeAttr::get(builder.getContext(), DT));

        op.setDeviceTypesAttr(
            mlir::ArrayAttr::get(builder.getContext(), deviceTypes));
      } else if constexpr (isOneOfTypes<Op, SetOp>) {
        assert(attrData.deviceTypeArchs.size() <= 1 &&
               "Set can only have a single architecture");
        if (!attrData.deviceTypeArchs.empty())
          op.setDeviceType(attrData.deviceTypeArchs[0]);
      } else {
        cgm.errorNYI(dirLoc, "OpenACC 'device_type' clause lowering for ",
                     dirKind);
      }
    }
  }
};

} // namespace

template <typename Op, typename TermOp>
mlir::LogicalResult CIRGenFunction::emitOpenACCOpAssociatedStmt(
    mlir::Location start, mlir::Location end, OpenACCDirectiveKind dirKind,
    SourceLocation dirLoc, llvm::ArrayRef<const OpenACCClause *> clauses,
    const Stmt *associatedStmt) {
  mlir::LogicalResult res = mlir::success();

  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;

  // Clause-emitter must be here because it might modify operands.
  OpenACCClauseCIREmitter clauseEmitter(getCIRGenModule(), dirKind, dirLoc);
  clauseEmitter.VisitClauseList(clauses);

  auto op = builder.create<Op>(start, retTy, operands);

  // Apply the attributes derived from the clauses.
  clauseEmitter.applyAttributes(builder, op);

  mlir::Block &block = op.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);

  LexicalScope ls{*this, start, builder.getInsertionBlock()};
  res = emitStmt(associatedStmt, /*useCurrentScope=*/true);

  builder.create<TermOp>(end);
  return res;
}

template <typename Op>
mlir::LogicalResult CIRGenFunction::emitOpenACCOp(
    mlir::Location start, OpenACCDirectiveKind dirKind, SourceLocation dirLoc,
    llvm::ArrayRef<const OpenACCClause *> clauses) {
  mlir::LogicalResult res = mlir::success();

  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;

  // Clause-emitter must be here because it might modify operands.
  OpenACCClauseCIREmitter clauseEmitter(getCIRGenModule(), dirKind, dirLoc);
  clauseEmitter.VisitClauseList(clauses);

  auto op = builder.create<Op>(start, retTy, operands);
  // Apply the attributes derived from the clauses.
  clauseEmitter.applyAttributes(builder, op);
  return res;
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCComputeConstruct(const OpenACCComputeConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  mlir::Location end = getLoc(s.getSourceRange().getEnd());

  switch (s.getDirectiveKind()) {
  case OpenACCDirectiveKind::Parallel:
    return emitOpenACCOpAssociatedStmt<ParallelOp, mlir::acc::YieldOp>(
        start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
        s.getStructuredBlock());
  case OpenACCDirectiveKind::Serial:
    return emitOpenACCOpAssociatedStmt<SerialOp, mlir::acc::YieldOp>(
        start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
        s.getStructuredBlock());
  case OpenACCDirectiveKind::Kernels:
    return emitOpenACCOpAssociatedStmt<KernelsOp, mlir::acc::TerminatorOp>(
        start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
        s.getStructuredBlock());
  default:
    llvm_unreachable("invalid compute construct kind");
  }
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCDataConstruct(const OpenACCDataConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  mlir::Location end = getLoc(s.getSourceRange().getEnd());

  return emitOpenACCOpAssociatedStmt<DataOp, mlir::acc::TerminatorOp>(
      start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
      s.getStructuredBlock());
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCInitConstruct(const OpenACCInitConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  return emitOpenACCOp<InitOp>(start, s.getDirectiveKind(), s.getDirectiveLoc(),
                               s.clauses());
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCSetConstruct(const OpenACCSetConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  return emitOpenACCOp<SetOp>(start, s.getDirectiveKind(), s.getDirectiveLoc(),
                              s.clauses());
}

mlir::LogicalResult CIRGenFunction::emitOpenACCShutdownConstruct(
    const OpenACCShutdownConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  return emitOpenACCOp<ShutdownOp>(start, s.getDirectiveKind(),
                                   s.getDirectiveLoc(), s.clauses());
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCLoopConstruct(const OpenACCLoopConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Loop Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCCombinedConstruct(
    const OpenACCCombinedConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Combined Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCEnterDataConstruct(
    const OpenACCEnterDataConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC EnterData Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCExitDataConstruct(
    const OpenACCExitDataConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC ExitData Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCHostDataConstruct(
    const OpenACCHostDataConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC HostData Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCWaitConstruct(const OpenACCWaitConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Wait Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCUpdateConstruct(const OpenACCUpdateConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Update Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCAtomicConstruct(const OpenACCAtomicConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Atomic Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCCacheConstruct(const OpenACCCacheConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Cache Construct");
  return mlir::failure();
}
