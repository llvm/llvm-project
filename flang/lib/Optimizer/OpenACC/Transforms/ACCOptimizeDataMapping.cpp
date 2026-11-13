//===- ACCOptimizeDataMapping.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass changes an implicit copy to a copyin in two cases: the variable
// is a named constant, or it is not modified inside its compute region.
//
// Example transformation:
//
//   Before:
//     %c = acc.copyin varPtr(%x) dataClause(acc_copy) implicit(true)
//     acc.parallel dataOperands(%c) {
//       ... // %c is only read
//     }
//     acc.copyout accPtr(%c) to varPtr(%x) dataClause(acc_copy) implicit(true)
//
//   After:
//     %c = acc.copyin varPtr(%x) implicit(true)
//     acc.parallel dataOperands(%c) {
//       ...
//     }
//     acc.delete accPtr(%c) dataClause(acc_copyin) implicit(true)
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/OpenACC/Passes.h"
#include "flang/Optimizer/OpenACC/Support/FIROpenACCUtils.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace fir::acc {
#define GEN_PASS_DEF_ACCOPTIMIZEDATAMAPPING
#include "flang/Optimizer/OpenACC/Passes.h.inc"
} // namespace fir::acc

using namespace mlir;

namespace {

/// Returns true if `var` is a named constant. A named constant cannot be
/// defined, not even by a procedure it is passed to, so it is never modified.
/// INTENT(IN) is not enough to conclude this, since the dummy argument can be
/// forwarded to a procedure that modifies it, or be modified through a
/// pointer when it has the TARGET attribute.
/// TODO: Consider an OpenACC extension that also uses the INTENT(IN) attribute
/// to avoid copyout without the additional checking. See OpenACC-extensions.md.
static bool isNamedConstant(Value var) {
  auto varIface = dyn_cast_or_null<fir::FortranVariableOpInterface>(
      fir::acc::getOriginalDef(var, /*stripDeclare=*/false).getDefiningOp());
  return varIface && varIface.isParameter();
}

/// Returns the address of the data described by the descriptor `accVar`. The
/// operations created for it are appended to `tempOps`.
static Value getDescribedData(OpBuilder &builder, Value accVar,
                              SmallVectorImpl<Operation *> &tempOps) {
  Value box = accVar;
  if (fir::isBoxAddress(accVar.getType())) {
    auto load = fir::LoadOp::create(builder, accVar.getLoc(), accVar);
    tempOps.push_back(load);
    box = load;
  }
  auto boxAddr = fir::BoxAddrOp::create(builder, accVar.getLoc(), box);
  tempOps.push_back(boxAddr);
  return boxAddr;
}

/// Returns true if `computeConstructOp` may modify `accVar`. When `data` is
/// set, it is the address of the data a descriptor maps, which is checked too.
static bool mayModify(Operation *computeConstructOp, Value accVar, Value data,
                      AliasAnalysis &aliasAnalysis) {
  if (aliasAnalysis.getModRef(computeConstructOp, accVar).isMod())
    return true;
  return data && aliasAnalysis.getModRef(computeConstructOp, data).isMod();
}

/// Returns the single acc.parallel, acc.kernels, or acc.serial that lists
/// `accVar` as a data operand, or null when there is not exactly one.
static Operation *getSingleDataOperandOwner(Value accVar) {
  Operation *owner = nullptr;
  for (Operation *user : accVar.getUsers()) {
    if (!llvm::is_contained(acc::getDataOperands(user), accVar))
      continue;
    if (!isa<ACC_COMPUTE_CONSTRUCT_OPS>(user))
      return nullptr;
    if (owner && owner != user)
      return nullptr;
    owner = user;
  }
  return owner;
}

/// A copyin and the acc.copyout paired with it.
struct CopyCandidate {
  acc::CopyinOp copyinOp;
  acc::CopyoutOp copyoutOp;
};

/// Remaining copies that follow the first converted copyin are moved in front
/// of it, so the copies are issued first. This is done because this pass relies
/// on alias analysis, which takes advantage of Fortran aliasing rules: for
/// example, dummy arguments do not alias. However, if they did alias, then we
/// are now depending on the order ACCImplicitData decided to map, because a
/// copyin followed by a copy is not the same as a copy followed by a copyin
/// (since only the first data action maps, everything else updates the
/// counter). By issuing the copies before the copyins, we can ensure we do not
/// break anything even if we relied on the language rules to do the
/// optimization. The other operands keep their relative order.
static void moveCopiesBeforeConvertedCopyins(
    Operation *regionOp, const llvm::SmallPtrSetImpl<Operation *> &converted) {
  SmallVector<Value> operands(acc::getDataOperands(regionOp));
  auto isConverted = [&](Value value) {
    Operation *def = value.getDefiningOp();
    return def && converted.contains(def);
  };
  auto first = llvm::find_if(operands, isConverted);
  if (first == operands.end())
    return;

  SmallVector<Value> result(operands.begin(), first);
  SmallVector<Value> copies;
  SmallVector<Value> rest;
  for (Value value : llvm::make_range(first, operands.end())) {
    auto copyin = dyn_cast_or_null<acc::CopyinOp>(value.getDefiningOp());
    bool isCopy =
        copyin && acc::getDataClause(copyin) == acc::DataClause::acc_copy;
    (isCopy ? copies : rest).push_back(value);
  }
  result.append(copies);
  result.append(rest);
  acc::getMutableDataOperands(regionOp).assign(result);
}

class ACCOptimizeDataMapping
    : public fir::acc::impl::ACCOptimizeDataMappingBase<
          ACCOptimizeDataMapping> {
public:
  void runOnOperation() override {
    llvm::SmallVector<CopyCandidate> candidates;
    getOperation().walk([&](acc::CopyinOp copyinOp) {
      SmallVector<Operation *> exitOps =
          acc::getPairedDataExitOps(acc::getAccVar(copyinOp));
      if (exitOps.size() != 1)
        return;
      auto copyoutOp = dyn_cast<acc::CopyoutOp>(exitOps.front());
      // Only apply this optimization when the copy is implicit because those
      // are the ones that user did not specify. However, it seems reasonable
      // to apply to explicit copies as a generalization.
      if (!copyoutOp || !copyinOp.getImplicit() || !copyoutOp.getImplicit())
        return;
      candidates.push_back({copyinOp, copyoutOp});
    });

    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();

    llvm::SmallPtrSet<Operation *, 16> converted;
    llvm::SmallVector<Operation *> tempOps;
    for (CopyCandidate candidate : candidates) {
      acc::CopyinOp copyinOp = candidate.copyinOp;
      Value accVar = acc::getAccVar(copyinOp);

      // A named constant is never modified, whichever regions use it.
      if (!isNamedConstant(acc::getVar(copyinOp))) {
        Operation *regionOp = getSingleDataOperandOwner(accVar);
        if (!regionOp)
          continue;
        Value data;
        if (fir::isBoxAddressOrValue(accVar.getType())) {
          // To simplify the alias analysis query, we create a temporary access
          // to the descriptor data that gets erased afterwards.
          OpBuilder builder(copyinOp);
          builder.setInsertionPointAfter(copyinOp);
          data = getDescribedData(builder, accVar, tempOps);
        }
        if (mayModify(regionOp, accVar, data, aliasAnalysis))
          continue;
      }

      acc::CopyoutOp copyoutOp = candidate.copyoutOp;

      copyinOp.setDataClause(acc::DataClause::acc_copyin);
      converted.insert(copyinOp);
      // The exit is a delete so the structured data model stays correct. A
      // delete decrements the present counter and deletes the data only when
      // the counter reaches zero.
      OpBuilder builder(copyoutOp);
      acc::DeleteOp::create(builder, copyoutOp.getLoc(), copyoutOp.getAccVar(),
                            copyoutOp.getBounds(), copyoutOp.getAsyncOperands(),
                            copyoutOp.getAsyncOperandsDeviceTypeAttr(),
                            copyoutOp.getAsyncOnlyAttr(),
                            acc::DataClause::acc_copyin,
                            copyoutOp.getStructured(), copyoutOp.getImplicit(),
                            copyoutOp.getNameAttr());
      copyoutOp.erase();
    }

    for (Operation *op : llvm::reverse(tempOps))
      op->erase();

    // Each construct that lists a converted copyin moves its remaining copies
    // in front of its first converted copyin. This is always legal, even for
    // a derived type and its components. ACCImplicitData maps the
    // derived-type object before a component. A write to a component aliases
    // that object, so the object is not optimized and its copy stays before
    // the component. When the object and a component are both optimized, they
    // keep their relative order. A write to one component does not alias a
    // different component, but the object still stays in front of both.
    // A derived-type object may contain a pointer to a different object. A
    // deep copy of that target could be reordered relative to the object and
    // break the mapping order. This cannot happen now: only implicit copies
    // are optimized, and implicit deep copy is not enabled.
    llvm::SmallPtrSet<Operation *, 16> regions;
    for (Operation *copyin : converted) {
      Value accVar = acc::getAccVar(cast<acc::CopyinOp>(copyin));
      for (Operation *user : copyin->getUsers())
        if (llvm::is_contained(acc::getDataOperands(user), accVar))
          regions.insert(user);
    }
    for (Operation *regionOp : regions)
      moveCopiesBeforeConvertedCopyins(regionOp, converted);
  }
};

} // namespace
