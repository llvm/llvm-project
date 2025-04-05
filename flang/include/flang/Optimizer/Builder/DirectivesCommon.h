//===-- DirectivesCommon.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
///
/// A location to place directive utilities shared across multiple lowering
/// and optimizer files, e.g. utilities shared in OpenMP and OpenACC.
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_DIRECTIVESCOMMON_H_
#define FORTRAN_OPTIMIZER_BUILDER_DIRECTIVESCOMMON_H_

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

namespace fir::factory {

/// Information gathered to generate bounds operation and data entry/exit
/// operations.
struct AddrAndBoundsInfo {
  explicit AddrAndBoundsInfo() {}
  explicit AddrAndBoundsInfo(mlir::Value addr, mlir::Value rawInput)
      : addr(addr), rawInput(rawInput) {}
  explicit AddrAndBoundsInfo(mlir::Value addr, mlir::Value rawInput,
                             mlir::Value isPresent)
      : addr(addr), rawInput(rawInput), isPresent(isPresent) {}
  explicit AddrAndBoundsInfo(mlir::Value addr, mlir::Value rawInput,
                             mlir::Value isPresent, mlir::Type boxType)
      : addr(addr), rawInput(rawInput), isPresent(isPresent), boxType(boxType) {
  }
  mlir::Value addr = nullptr;
  mlir::Value rawInput = nullptr;
  mlir::Value isPresent = nullptr;
  mlir::Type boxType = nullptr;
  void dump(llvm::raw_ostream &os) {
    os << "AddrAndBoundsInfo addr: " << addr << "\n";
    os << "AddrAndBoundsInfo rawInput: " << rawInput << "\n";
    os << "AddrAndBoundsInfo isPresent: " << isPresent << "\n";
    os << "AddrAndBoundsInfo boxType: " << boxType << "\n";
  }
};

inline AddrAndBoundsInfo getDataOperandBaseAddr(fir::FirOpBuilder &builder,
                                                mlir::Value symAddr,
                                                bool isOptional,
                                                mlir::Location loc,
                                                bool unwrapFirBox = true) {
  mlir::Value rawInput = symAddr;
  if (auto declareOp =
          mlir::dyn_cast_or_null<hlfir::DeclareOp>(symAddr.getDefiningOp())) {
    symAddr = declareOp.getResults()[0];
    rawInput = declareOp.getResults()[1];
  }

  if (!symAddr)
    llvm::report_fatal_error("could not retrieve symbol address");

  mlir::Value isPresent;
  if (isOptional)
    isPresent =
        builder.create<fir::IsPresentOp>(loc, builder.getI1Type(), rawInput);

  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(
          fir::unwrapRefType(symAddr.getType()))) {
    if (mlir::isa<fir::RecordType>(boxTy.getEleTy()))
      TODO(loc, "derived type");

    // In case of a box reference, load it here to get the box value.
    // This is preferrable because then the same box value can then be used for
    // all address/dimension retrievals. For Fortran optional though, leave
    // the load generation for later so it can be done in the appropriate
    // if branches.
    if (unwrapFirBox && mlir::isa<fir::ReferenceType>(symAddr.getType()) &&
        !isOptional) {
      mlir::Value addr = builder.create<fir::LoadOp>(loc, symAddr);
      return AddrAndBoundsInfo(addr, rawInput, isPresent, boxTy);
    }

    return AddrAndBoundsInfo(symAddr, rawInput, isPresent, boxTy);
  }
  return AddrAndBoundsInfo(symAddr, rawInput, isPresent);
}

template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
gatherBoundsOrBoundValues(fir::FirOpBuilder &builder, mlir::Location loc,
                          fir::ExtendedValue dataExv, mlir::Value box,
                          bool collectValuesOnly = false) {
  assert(box && "box must exist");
  llvm::SmallVector<mlir::Value> values;
  mlir::Value byteStride;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<BoundsType>();
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  for (unsigned dim = 0; dim < dataExv.rank(); ++dim) {
    mlir::Value d = builder.createIntegerConstant(loc, idxTy, dim);
    mlir::Value baseLb =
        fir::factory::readLowerBound(builder, loc, dataExv, dim, one);
    auto dimInfo =
        builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, d);
    mlir::Value lb = builder.createIntegerConstant(loc, idxTy, 0);
    mlir::Value ub =
        builder.create<mlir::arith::SubIOp>(loc, dimInfo.getExtent(), one);
    if (dim == 0) // First stride is the element size.
      byteStride = dimInfo.getByteStride();
    if (collectValuesOnly) {
      values.push_back(lb);
      values.push_back(ub);
      values.push_back(dimInfo.getExtent());
      values.push_back(byteStride);
      values.push_back(baseLb);
    } else {
      mlir::Value bound = builder.create<BoundsOp>(
          loc, boundTy, lb, ub, dimInfo.getExtent(), byteStride, true, baseLb);
      values.push_back(bound);
    }
    // Compute the stride for the next dimension.
    byteStride = builder.create<mlir::arith::MulIOp>(loc, byteStride,
                                                     dimInfo.getExtent());
  }
  return values;
}

/// Generate the bounds operation from the descriptor information.
template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
genBoundsOpsFromBox(fir::FirOpBuilder &builder, mlir::Location loc,
                    fir::ExtendedValue dataExv, AddrAndBoundsInfo &info) {
  llvm::SmallVector<mlir::Value> bounds;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<BoundsType>();

  assert(mlir::isa<fir::BaseBoxType>(info.boxType) &&
         "expect fir.box or fir.class");
  assert(fir::unwrapRefType(info.addr.getType()) == info.boxType &&
         "expected box type consistency");

  if (info.isPresent) {
    llvm::SmallVector<mlir::Type> resTypes;
    constexpr unsigned nbValuesPerBound = 5;
    for (unsigned dim = 0; dim < dataExv.rank() * nbValuesPerBound; ++dim)
      resTypes.push_back(idxTy);

    mlir::Operation::result_range ifRes =
        builder.genIfOp(loc, resTypes, info.isPresent, /*withElseRegion=*/true)
            .genThen([&]() {
              mlir::Value box =
                  !fir::isBoxAddress(info.addr.getType())
                      ? info.addr
                      : builder.create<fir::LoadOp>(loc, info.addr);
              llvm::SmallVector<mlir::Value> boundValues =
                  gatherBoundsOrBoundValues<BoundsOp, BoundsType>(
                      builder, loc, dataExv, box,
                      /*collectValuesOnly=*/true);
              builder.create<fir::ResultOp>(loc, boundValues);
            })
            .genElse([&] {
              // Box is not present. Populate bound values with default values.
              llvm::SmallVector<mlir::Value> boundValues;
              mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
              mlir::Value mOne = builder.createMinusOneInteger(loc, idxTy);
              for (unsigned dim = 0; dim < dataExv.rank(); ++dim) {
                boundValues.push_back(zero); // lb
                boundValues.push_back(mOne); // ub
                boundValues.push_back(zero); // extent
                boundValues.push_back(zero); // byteStride
                boundValues.push_back(zero); // baseLb
              }
              builder.create<fir::ResultOp>(loc, boundValues);
            })
            .getResults();
    // Create the bound operations outside the if-then-else with the if op
    // results.
    for (unsigned i = 0; i < ifRes.size(); i += nbValuesPerBound) {
      mlir::Value bound = builder.create<BoundsOp>(
          loc, boundTy, ifRes[i], ifRes[i + 1], ifRes[i + 2], ifRes[i + 3],
          true, ifRes[i + 4]);
      bounds.push_back(bound);
    }
  } else {
    mlir::Value box = !fir::isBoxAddress(info.addr.getType())
                          ? info.addr
                          : builder.create<fir::LoadOp>(loc, info.addr);
    bounds = gatherBoundsOrBoundValues<BoundsOp, BoundsType>(builder, loc,
                                                             dataExv, box);
  }
  return bounds;
}

/// Generate bounds operation for base array without any subscripts
/// provided.
template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
genBaseBoundsOps(fir::FirOpBuilder &builder, mlir::Location loc,
                 fir::ExtendedValue dataExv, bool isAssumedSize) {
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<BoundsType>();
  llvm::SmallVector<mlir::Value> bounds;

  if (dataExv.rank() == 0)
    return bounds;

  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  const unsigned rank = dataExv.rank();
  for (unsigned dim = 0; dim < rank; ++dim) {
    mlir::Value baseLb =
        fir::factory::readLowerBound(builder, loc, dataExv, dim, one);
    mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
    mlir::Value ub;
    mlir::Value lb = zero;
    mlir::Value ext = fir::factory::readExtent(builder, loc, dataExv, dim);
    if (isAssumedSize && dim + 1 == rank) {
      ext = zero;
      ub = lb;
    } else {
      // ub = extent - 1
      ub = builder.create<mlir::arith::SubIOp>(loc, ext, one);
    }

    mlir::Value bound =
        builder.create<BoundsOp>(loc, boundTy, lb, ub, ext, one, false, baseLb);
    bounds.push_back(bound);
  }
  return bounds;
}

template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
genImplicitBoundsOps(fir::FirOpBuilder &builder, AddrAndBoundsInfo &info,
                     fir::ExtendedValue dataExv, bool dataExvIsAssumedSize,
                     mlir::Location loc) {
  llvm::SmallVector<mlir::Value> bounds;

  mlir::Value baseOp = info.rawInput;
  if (mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(baseOp.getType())))
    bounds =
        genBoundsOpsFromBox<BoundsOp, BoundsType>(builder, loc, dataExv, info);
  if (mlir::isa<fir::SequenceType>(fir::unwrapRefType(baseOp.getType()))) {
    bounds = genBaseBoundsOps<BoundsOp, BoundsType>(builder, loc, dataExv,
                                                    dataExvIsAssumedSize);
  }

  return bounds;
}

} // namespace fir::factory
#endif // FORTRAN_OPTIMIZER_BUILDER_DIRECTIVESCOMMON_H_
