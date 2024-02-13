//===-- Optimizer/Support/Utils.h -------------------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_SUPPORT_UTILS_H
#define FORTRAN_OPTIMIZER_SUPPORT_UTILS_H

#include "flang/Common/default-kinds.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace fir {
/// Return the integer value of a arith::ConstantOp.
inline std::int64_t toInt(mlir::arith::ConstantOp cop) {
  return cop.getValue().cast<mlir::IntegerAttr>().getValue().getSExtValue();
}

// Reconstruct binding tables for dynamic dispatch.
using BindingTable = llvm::DenseMap<llvm::StringRef, unsigned>;
using BindingTables = llvm::DenseMap<llvm::StringRef, BindingTable>;

inline void buildBindingTables(BindingTables &bindingTables,
                               mlir::ModuleOp mod) {

  // The binding tables are defined in FIR after lowering inside fir.type_info
  // operations. Go through each binding tables and store the procedure name and
  // binding index for later use by the fir.dispatch conversion pattern.
  for (auto typeInfo : mod.getOps<fir::TypeInfoOp>()) {
    unsigned bindingIdx = 0;
    BindingTable bindings;
    if (typeInfo.getDispatchTable().empty()) {
      bindingTables[typeInfo.getSymName()] = bindings;
      continue;
    }
    for (auto dtEntry :
         typeInfo.getDispatchTable().front().getOps<fir::DTEntryOp>()) {
      bindings[dtEntry.getMethod()] = bindingIdx;
      ++bindingIdx;
    }
    bindingTables[typeInfo.getSymName()] = bindings;
  }
}

// Translate front-end KINDs for use in the IR and code gen.
inline std::vector<fir::KindTy>
fromDefaultKinds(const Fortran::common::IntrinsicTypeDefaultKinds &defKinds) {
  return {static_cast<fir::KindTy>(defKinds.GetDefaultKind(
              Fortran::common::TypeCategory::Character)),
          static_cast<fir::KindTy>(
              defKinds.GetDefaultKind(Fortran::common::TypeCategory::Complex)),
          static_cast<fir::KindTy>(defKinds.doublePrecisionKind()),
          static_cast<fir::KindTy>(
              defKinds.GetDefaultKind(Fortran::common::TypeCategory::Integer)),
          static_cast<fir::KindTy>(
              defKinds.GetDefaultKind(Fortran::common::TypeCategory::Logical)),
          static_cast<fir::KindTy>(
              defKinds.GetDefaultKind(Fortran::common::TypeCategory::Real))};
}

inline std::string mlirTypeToString(mlir::Type type) {
  std::string result{};
  llvm::raw_string_ostream sstream(result);
  sstream << type;
  return result;
}

inline std::string numericMlirTypeToFortran(fir::FirOpBuilder &builder,
                                            mlir::Type type, mlir::Location loc,
                                            const llvm::Twine &name) {
  if (type.isF16())
    return "REAL(KIND=2)";
  else if (type.isBF16())
    return "REAL(KIND=3)";
  else if (type.isTF32())
    return "REAL(KIND=unknown)";
  else if (type.isF32())
    return "REAL(KIND=4)";
  else if (type.isF64())
    return "REAL(KIND=8)";
  else if (type.isF80())
    return "REAL(KIND=10)";
  else if (type.isF128())
    return "REAL(KIND=16)";
  else if (type.isInteger(8))
    return "INTEGER(KIND=1)";
  else if (type.isInteger(16))
    return "INTEGER(KIND=2)";
  else if (type.isInteger(32))
    return "INTEGER(KIND=4)";
  else if (type.isInteger(64))
    return "INTEGER(KIND=8)";
  else if (type.isInteger(128))
    return "INTEGER(KIND=16)";
  else if (type == fir::ComplexType::get(builder.getContext(), 2))
    return "COMPLEX(KIND=2)";
  else if (type == fir::ComplexType::get(builder.getContext(), 3))
    return "COMPLEX(KIND=3)";
  else if (type == fir::ComplexType::get(builder.getContext(), 4))
    return "COMPLEX(KIND=4)";
  else if (type == fir::ComplexType::get(builder.getContext(), 8))
    return "COMPLEX(KIND=8)";
  else if (type == fir::ComplexType::get(builder.getContext(), 10))
    return "COMPLEX(KIND=10)";
  else if (type == fir::ComplexType::get(builder.getContext(), 16))
    return "COMPLEX(KIND=16)";
  else
    fir::emitFatalError(loc, "unsupported type in " + name + ": " +
                                 fir::mlirTypeToString(type));
}

inline void intrinsicTypeTODO(fir::FirOpBuilder &builder, mlir::Type type,
                              mlir::Location loc,
                              const llvm::Twine &intrinsicName) {
  TODO(loc,
       "intrinsic: " +
           fir::numericMlirTypeToFortran(builder, type, loc, intrinsicName) +
           " in " + intrinsicName);
}

using MinlocBodyOpGeneratorTy = llvm::function_ref<mlir::Value(
    fir::FirOpBuilder &, mlir::Location, const mlir::Type &, mlir::Value,
    mlir::Value, mlir::Value, const llvm::SmallVectorImpl<mlir::Value> &)>;
using InitValGeneratorTy = llvm::function_ref<mlir::Value(
    fir::FirOpBuilder &, mlir::Location, const mlir::Type &)>;
using AddrGeneratorTy = llvm::function_ref<mlir::Value(
    fir::FirOpBuilder &, mlir::Location, const mlir::Type &, mlir::Value,
    mlir::Value)>;

// Produces a loop nest for a Minloc intrinsic.
inline void genMinMaxlocReductionLoop(
    fir::FirOpBuilder &builder, mlir::Value array,
    fir::InitValGeneratorTy initVal, fir::MinlocBodyOpGeneratorTy genBody,
    fir::AddrGeneratorTy getAddrFn, unsigned rank, mlir::Type elementType,
    mlir::Location loc, mlir::Type maskElemType, mlir::Value resultArr,
    bool maskMayBeLogicalScalar) {
  mlir::IndexType idxTy = builder.getIndexType();

  mlir::Value zeroIdx = builder.createIntegerConstant(loc, idxTy, 0);

  fir::SequenceType::Shape flatShape(rank,
                                     fir::SequenceType::getUnknownExtent());
  mlir::Type arrTy = fir::SequenceType::get(flatShape, elementType);
  mlir::Type boxArrTy = fir::BoxType::get(arrTy);
  array = builder.create<fir::ConvertOp>(loc, boxArrTy, array);

  mlir::Type resultElemType = hlfir::getFortranElementType(resultArr.getType());
  mlir::Value flagSet = builder.createIntegerConstant(loc, resultElemType, 1);
  mlir::Value zero = builder.createIntegerConstant(loc, resultElemType, 0);
  mlir::Value flagRef = builder.createTemporary(loc, resultElemType);
  builder.create<fir::StoreOp>(loc, zero, flagRef);

  mlir::Value init = initVal(builder, loc, elementType);
  llvm::SmallVector<mlir::Value, Fortran::common::maxRank> bounds;

  assert(rank > 0 && "rank cannot be zero");
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);

  // Compute all the upper bounds before the loop nest.
  // It is not strictly necessary for performance, since the loop nest
  // does not have any store operations and any LICM optimization
  // should be able to optimize the redundancy.
  for (unsigned i = 0; i < rank; ++i) {
    mlir::Value dimIdx = builder.createIntegerConstant(loc, idxTy, i);
    auto dims =
        builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, array, dimIdx);
    mlir::Value len = dims.getResult(1);
    // We use C indexing here, so len-1 as loopcount
    mlir::Value loopCount = builder.create<mlir::arith::SubIOp>(loc, len, one);
    bounds.push_back(loopCount);
  }
  // Create a loop nest consisting of OP operations.
  // Collect the loops' induction variables into indices array,
  // which will be used in the innermost loop to load the input
  // array's element.
  // The loops are generated such that the innermost loop processes
  // the 0 dimension.
  llvm::SmallVector<mlir::Value, Fortran::common::maxRank> indices;
  for (unsigned i = rank; 0 < i; --i) {
    mlir::Value step = one;
    mlir::Value loopCount = bounds[i - 1];
    auto loop =
        builder.create<fir::DoLoopOp>(loc, zeroIdx, loopCount, step, false,
                                      /*finalCountValue=*/false, init);
    init = loop.getRegionIterArgs()[0];
    indices.push_back(loop.getInductionVar());
    // Set insertion point to the loop body so that the next loop
    // is inserted inside the current one.
    builder.setInsertionPointToStart(loop.getBody());
  }

  // Reverse the indices such that they are ordered as:
  //   <dim-0-idx, dim-1-idx, ...>
  std::reverse(indices.begin(), indices.end());
  mlir::Value reductionVal =
      genBody(builder, loc, elementType, array, flagRef, init, indices);

  // Unwind the loop nest and insert ResultOp on each level
  // to return the updated value of the reduction to the enclosing
  // loops.
  for (unsigned i = 0; i < rank; ++i) {
    auto result = builder.create<fir::ResultOp>(loc, reductionVal);
    // Proceed to the outer loop.
    auto loop = mlir::cast<fir::DoLoopOp>(result->getParentOp());
    reductionVal = loop.getResult(0);
    // Set insertion point after the loop operation that we have
    // just processed.
    builder.setInsertionPointAfter(loop.getOperation());
  }
  // End of loop nest. The insertion point is after the outermost loop.
  if (maskMayBeLogicalScalar) {
    if (fir::IfOp ifOp =
            mlir::dyn_cast<fir::IfOp>(builder.getBlock()->getParentOp())) {
      builder.create<fir::ResultOp>(loc, reductionVal);
      builder.setInsertionPointAfter(ifOp);
      // Redefine flagSet to escape scope of ifOp
      flagSet = builder.createIntegerConstant(loc, resultElemType, 1);
      reductionVal = ifOp.getResult(0);
    }
  }

  // Check for case where array was full of max values.
  // flag will be 0 if mask was never true, 1 if mask was true as some point,
  // this is needed to avoid catching cases where we didn't access any elements
  // e.g. mask=.FALSE.
  mlir::Value flagValue =
      builder.create<fir::LoadOp>(loc, resultElemType, flagRef);
  mlir::Value flagCmp = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, flagValue, flagSet);
  fir::IfOp ifMaskTrueOp =
      builder.create<fir::IfOp>(loc, flagCmp, /*withElseRegion=*/false);
  builder.setInsertionPointToStart(&ifMaskTrueOp.getThenRegion().front());

  mlir::Value testInit = initVal(builder, loc, elementType);
  fir::IfOp ifMinSetOp;
  if (elementType.isa<mlir::FloatType>()) {
    mlir::Value cmp = builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OEQ, testInit, reductionVal);
    ifMinSetOp = builder.create<fir::IfOp>(loc, cmp,
                                           /*withElseRegion*/ false);
  } else {
    mlir::Value cmp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, testInit, reductionVal);
    ifMinSetOp = builder.create<fir::IfOp>(loc, cmp,
                                           /*withElseRegion*/ false);
  }
  builder.setInsertionPointToStart(&ifMinSetOp.getThenRegion().front());

  // Load output array with 1s instead of 0s
  for (unsigned int i = 0; i < rank; ++i) {
    mlir::Value index = builder.createIntegerConstant(loc, idxTy, i);
    mlir::Value resultElemAddr =
        getAddrFn(builder, loc, resultElemType, resultArr, index);
    builder.create<fir::StoreOp>(loc, flagSet, resultElemAddr);
  }
  builder.setInsertionPointAfter(ifMaskTrueOp);
}

inline fir::CUDADataAttributeAttr
getCUDADataAttribute(mlir::MLIRContext *mlirContext,
                     std::optional<Fortran::common::CUDADataAttr> cudaAttr) {
  if (cudaAttr) {
    fir::CUDADataAttribute attr;
    switch (*cudaAttr) {
    case Fortran::common::CUDADataAttr::Constant:
      attr = fir::CUDADataAttribute::Constant;
      break;
    case Fortran::common::CUDADataAttr::Device:
      attr = fir::CUDADataAttribute::Device;
      break;
    case Fortran::common::CUDADataAttr::Managed:
      attr = fir::CUDADataAttribute::Managed;
      break;
    case Fortran::common::CUDADataAttr::Pinned:
      attr = fir::CUDADataAttribute::Pinned;
      break;
    case Fortran::common::CUDADataAttr::Shared:
      attr = fir::CUDADataAttribute::Shared;
      break;
    case Fortran::common::CUDADataAttr::Texture:
      // Obsolete attribute
      return {};
    }
    return fir::CUDADataAttributeAttr::get(mlirContext, attr);
  }
  return {};
}

inline fir::CUDAProcAttributeAttr getCUDAProcAttribute(
    mlir::MLIRContext *mlirContext,
    std::optional<Fortran::common::CUDASubprogramAttrs> cudaAttr) {
  if (cudaAttr) {
    fir::CUDAProcAttribute attr;
    switch (*cudaAttr) {
    case Fortran::common::CUDASubprogramAttrs::Host:
      attr = fir::CUDAProcAttribute::Host;
      break;
    case Fortran::common::CUDASubprogramAttrs::Device:
      attr = fir::CUDAProcAttribute::Device;
      break;
    case Fortran::common::CUDASubprogramAttrs::HostDevice:
      attr = fir::CUDAProcAttribute::HostDevice;
      break;
    case Fortran::common::CUDASubprogramAttrs::Global:
      attr = fir::CUDAProcAttribute::Global;
      break;
    case Fortran::common::CUDASubprogramAttrs::Grid_Global:
      attr = fir::CUDAProcAttribute::GridGlobal;
      break;
    }
    return fir::CUDAProcAttributeAttr::get(mlirContext, attr);
  }
  return {};
}

} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_UTILS_H
