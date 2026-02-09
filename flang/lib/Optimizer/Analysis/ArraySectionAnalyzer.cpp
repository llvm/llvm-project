//===- ArraySectionAnalyzer.cpp - Analyze array sections ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/ArraySectionAnalyzer.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "array-section-analyzer"

using namespace fir;

ArraySectionAnalyzer::SectionDesc::SectionDesc(mlir::Value lb, mlir::Value ub,
                                               mlir::Value stride)
    : lb(lb), ub(ub), stride(stride) {
  assert(lb && "lower bound or index must be specified");
  normalize();
}

void ArraySectionAnalyzer::SectionDesc::normalize() {
  if (!ub)
    ub = lb;
  if (lb == ub)
    stride = nullptr;
  if (stride)
    if (auto val = fir::getIntIfConstant(stride))
      if (*val == 1)
        stride = nullptr;
}

bool ArraySectionAnalyzer::SectionDesc::operator==(
    const SectionDesc &other) const {
  return lb == other.lb && ub == other.ub && stride == other.stride;
}

ArraySectionAnalyzer::SectionDesc
ArraySectionAnalyzer::readSectionDesc(mlir::Operation::operand_iterator &it,
                                      bool isTriplet) {
  if (isTriplet)
    return {*it++, *it++, *it++};
  return {*it++, nullptr, nullptr};
}

std::pair<mlir::Value, mlir::Value>
ArraySectionAnalyzer::getOrderedBounds(const SectionDesc &desc) {
  mlir::Value stride = desc.stride;
  // Null stride means stride=1.
  if (!stride)
    return {desc.lb, desc.ub};
  // Reverse the bounds, if stride is negative.
  if (auto val = fir::getIntIfConstant(stride)) {
    if (*val >= 0)
      return {desc.lb, desc.ub};
    else
      return {desc.ub, desc.lb};
  }

  return {nullptr, nullptr};
}

bool ArraySectionAnalyzer::areDisjointSections(const SectionDesc &desc1,
                                               const SectionDesc &desc2) {
  auto [lb1, ub1] = getOrderedBounds(desc1);
  auto [lb2, ub2] = getOrderedBounds(desc2);
  if (!lb1 || !lb2)
    return false;
  // Note that this comparison must be made on the ordered bounds,
  // otherwise 'a(x:y:1) = a(z:x-1:-1) + 1' may be incorrectly treated
  // as not overlapping (x=2, y=10, z=9).
  if (isLess(ub1, lb2) || isLess(ub2, lb1))
    return true;
  return false;
}

bool ArraySectionAnalyzer::areIdenticalSections(const SectionDesc &desc1,
                                                const SectionDesc &desc2) {
  if (desc1 == desc2)
    return true;
  return false;
}

ArraySectionAnalyzer::SlicesOverlapKind
ArraySectionAnalyzer::analyze(mlir::Value ref1, mlir::Value ref2) {
  if (ref1 == ref2)
    return SlicesOverlapKind::DefinitelyIdentical;

  auto des1 = ref1.getDefiningOp<hlfir::DesignateOp>();
  auto des2 = ref2.getDefiningOp<hlfir::DesignateOp>();
  // We only support a pair of designators right now.
  if (!des1 || !des2)
    return SlicesOverlapKind::Unknown;

  if (des1.getMemref() != des2.getMemref()) {
    // If the bases are different, then there is unknown overlap.
    LLVM_DEBUG(llvm::dbgs() << "No identical base for:\n"
                            << des1 << "and:\n"
                            << des2 << "\n");
    return SlicesOverlapKind::Unknown;
  }

  // Require all components of the designators to be the same.
  // It might be too strict, e.g. we may probably allow for
  // different type parameters.
  if (des1.getComponent() != des2.getComponent() ||
      des1.getComponentShape() != des2.getComponentShape() ||
      des1.getSubstring() != des2.getSubstring() ||
      des1.getComplexPart() != des2.getComplexPart() ||
      des1.getTypeparams() != des2.getTypeparams()) {
    LLVM_DEBUG(llvm::dbgs() << "Different designator specs for:\n"
                            << des1 << "and:\n"
                            << des2 << "\n");
    return SlicesOverlapKind::Unknown;
  }

  // Analyze the subscripts.
  auto des1It = des1.getIndices().begin();
  auto des2It = des2.getIndices().begin();
  bool identicalTriplets = true;
  bool identicalIndices = true;
  for (auto [isTriplet1, isTriplet2] :
       llvm::zip(des1.getIsTriplet(), des2.getIsTriplet())) {
    SectionDesc desc1 = readSectionDesc(des1It, isTriplet1);
    SectionDesc desc2 = readSectionDesc(des2It, isTriplet2);

    // See if we can prove that any of the sections do not overlap.
    // This is mostly a Polyhedron/nf performance hack that looks for
    // particular relations between the lower and upper bounds
    // of the array sections, e.g. for any positive constant C:
    //   X:Y does not overlap with (Y+C):Z
    //   X:Y does not overlap with Z:(X-C)
    if (areDisjointSections(desc1, desc2))
      return SlicesOverlapKind::DefinitelyDisjoint;

    if (!areIdenticalSections(desc1, desc2)) {
      if (isTriplet1 || isTriplet2) {
        // For example:
        //   hlfir.designate %6#0 (%c2:%c7999:%c1, %c1:%c120:%c1, %0)
        //   hlfir.designate %6#0 (%c2:%c7999:%c1, %c1:%c120:%c1, %1)
        //
        // If all the triplets (section speficiers) are the same, then
        // we do not care if %0 is equal to %1 - the slices are either
        // identical or completely disjoint.
        //
        // Also, treat these as identical sections:
        //   hlfir.designate %6#0 (%c2:%c2:%c1)
        //   hlfir.designate %6#0 (%c2)
        identicalTriplets = false;
        LLVM_DEBUG(llvm::dbgs() << "Triplet mismatch for:\n"
                                << des1 << "and:\n"
                                << des2 << "\n");
      } else {
        identicalIndices = false;
        LLVM_DEBUG(llvm::dbgs() << "Indices mismatch for:\n"
                                << des1 << "and:\n"
                                << des2 << "\n");
      }
    }
  }

  if (identicalTriplets) {
    if (identicalIndices)
      return SlicesOverlapKind::DefinitelyIdentical;
    else
      return SlicesOverlapKind::EitherIdenticalOrDisjoint;
  }

  LLVM_DEBUG(llvm::dbgs() << "Different sections for:\n"
                          << des1 << "and:\n"
                          << des2 << "\n");
  return SlicesOverlapKind::Unknown;
}

bool ArraySectionAnalyzer::isLess(mlir::Value v1, mlir::Value v2) {
  auto removeConvert = [](mlir::Value v) -> mlir::Operation * {
    auto *op = v.getDefiningOp();
    while (auto conv = mlir::dyn_cast_or_null<fir::ConvertOp>(op))
      op = conv.getValue().getDefiningOp();
    return op;
  };

  auto isPositiveConstant = [](mlir::Value v) -> bool {
    if (auto val = fir::getIntIfConstant(v))
      return *val > 0;
    return false;
  };

  auto *op1 = removeConvert(v1);
  auto *op2 = removeConvert(v2);
  if (!op1 || !op2)
    return false;

  // Check if they are both constants.
  if (auto val1 = fir::getIntIfConstant(op1->getResult(0)))
    if (auto val2 = fir::getIntIfConstant(op2->getResult(0)))
      return *val1 < *val2;

  // Handle some variable cases (C > 0):
  //   v2 = v1 + C
  //   v2 = C + v1
  //   v1 = v2 - C
  if (auto addi = mlir::dyn_cast<mlir::arith::AddIOp>(op2))
    if ((addi.getLhs().getDefiningOp() == op1 &&
         isPositiveConstant(addi.getRhs())) ||
        (addi.getRhs().getDefiningOp() == op1 &&
         isPositiveConstant(addi.getLhs())))
      return true;
  if (auto subi = mlir::dyn_cast<mlir::arith::SubIOp>(op1))
    if (subi.getLhs().getDefiningOp() == op2 &&
        isPositiveConstant(subi.getRhs()))
      return true;
  return false;
}

/// Returns the array indices for the given hlfir.designate.
/// It recognizes the computations used to transform the one-based indices
/// into the array's lb-based indices, and returns the one-based indices
/// in these cases.
static llvm::SmallVector<mlir::Value>
getDesignatorIndices(hlfir::DesignateOp designate) {
  mlir::Value memref = designate.getMemref();

  // If the object is a box, then the indices may be adjusted
  // according to the box's lower bound(s). Scan through
  // the computations to try to find the one-based indices.
  if (mlir::isa<fir::BaseBoxType>(memref.getType())) {
    // Look for the following pattern:
    //   %13 = fir.load %12 : !fir.ref<!fir.box<...>
    //   %14:3 = fir.box_dims %13, %c0 : (!fir.box<...>, index) -> ...
    //   %17 = arith.subi %14#0, %c1 : index
    //   %18 = arith.addi %arg2, %17 : index
    //   %19 = hlfir.designate %13 (%18)  : (!fir.box<...>, index) -> ...
    //
    // %arg2 is a one-based index.

    auto isNormalizedLb = [memref](mlir::Value v, unsigned dim) {
      // Return true, if v and dim are such that:
      //   %14:3 = fir.box_dims %13, %dim : (!fir.box<...>, index) -> ...
      //   %17 = arith.subi %14#0, %c1 : index
      //   %19 = hlfir.designate %13 (...)  : (!fir.box<...>, index) -> ...
      if (auto subOp =
              mlir::dyn_cast_or_null<mlir::arith::SubIOp>(v.getDefiningOp())) {
        auto cst = fir::getIntIfConstant(subOp.getRhs());
        if (!cst || *cst != 1)
          return false;
        if (auto dimsOp = mlir::dyn_cast_or_null<fir::BoxDimsOp>(
                subOp.getLhs().getDefiningOp())) {
          if (memref != dimsOp.getVal() ||
              dimsOp.getResult(0) != subOp.getLhs())
            return false;
          auto dimsOpDim = fir::getIntIfConstant(dimsOp.getDim());
          return dimsOpDim && dimsOpDim == dim;
        }
      }
      return false;
    };

    llvm::SmallVector<mlir::Value> newIndices;
    for (auto index : llvm::enumerate(designate.getIndices())) {
      if (auto addOp = mlir::dyn_cast_or_null<mlir::arith::AddIOp>(
              index.value().getDefiningOp())) {
        for (unsigned opNum = 0; opNum < 2; ++opNum)
          if (isNormalizedLb(addOp->getOperand(opNum), index.index())) {
            newIndices.push_back(addOp->getOperand((opNum + 1) % 2));
            break;
          }

        // If new one-based index was not added, exit early.
        if (newIndices.size() <= index.index())
          break;
      }
    }

    // If any of the indices is not adjusted to the array's lb,
    // then return the original designator indices.
    if (newIndices.size() != designate.getIndices().size())
      return designate.getIndices();

    return newIndices;
  }

  return designate.getIndices();
}

bool fir::ArraySectionAnalyzer::isDesignatingArrayInOrder(
    hlfir::DesignateOp designate, hlfir::ElementalOpInterface elemental) {

  auto indices = getDesignatorIndices(designate);
  auto elementalIndices = elemental.getIndices();
  if (indices.size() == elementalIndices.size())
    return std::equal(indices.begin(), indices.end(), elementalIndices.begin(),
                      elementalIndices.end());
  return false;
}
