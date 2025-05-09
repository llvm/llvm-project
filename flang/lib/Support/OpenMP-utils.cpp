//===-- lib/Support/OpenMP-utils.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Support/OpenMP-utils.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/OpDefinition.h"

namespace Fortran::common::openmp {
mlir::Block *genEntryBlock(mlir::OpBuilder &builder, const EntryBlockArgs &args,
    mlir::Region &region) {
  assert(args.isValid() && "invalid args");
  assert(region.empty() && "non-empty region");

  llvm::SmallVector<mlir::Type> types;
  llvm::SmallVector<mlir::Location> locs;
  unsigned numVars = args.hasDeviceAddr.vars.size() + args.hostEvalVars.size() +
      args.inReduction.vars.size() + args.map.vars.size() +
      args.priv.vars.size() + args.reduction.vars.size() +
      args.taskReduction.vars.size() + args.useDeviceAddr.vars.size() +
      args.useDevicePtr.vars.size();
  types.reserve(numVars);
  locs.reserve(numVars);

  auto extractTypeLoc = [&types, &locs](llvm::ArrayRef<mlir::Value> vals) {
    llvm::transform(vals, std::back_inserter(types),
        [](mlir::Value v) { return v.getType(); });
    llvm::transform(vals, std::back_inserter(locs),
        [](mlir::Value v) { return v.getLoc(); });
  };

  // Populate block arguments in clause name alphabetical order to match
  // expected order by the BlockArgOpenMPOpInterface.
  extractTypeLoc(args.hasDeviceAddr.vars);
  extractTypeLoc(args.hostEvalVars);
  extractTypeLoc(args.inReduction.vars);
  extractTypeLoc(args.map.vars);
  extractTypeLoc(args.priv.vars);
  extractTypeLoc(args.reduction.vars);
  extractTypeLoc(args.taskReduction.vars);
  extractTypeLoc(args.useDeviceAddr.vars);
  extractTypeLoc(args.useDevicePtr.vars);

  return builder.createBlock(&region, {}, types, locs);
}

mlir::omp::MapInfoOp createMapInfoOp(mlir::OpBuilder &builder,
    mlir::Location loc, mlir::Value baseAddr, mlir::Value varPtrPtr,
    llvm::StringRef name, llvm::ArrayRef<mlir::Value> bounds,
    llvm::ArrayRef<mlir::Value> members, mlir::ArrayAttr membersIndex,
    uint64_t mapType, mlir::omp::VariableCaptureKind mapCaptureType,
    mlir::Type retTy, bool partialMap, mlir::FlatSymbolRefAttr mapperId) {

  if (auto boxTy = llvm::dyn_cast<fir::BaseBoxType>(baseAddr.getType())) {
    baseAddr = builder.create<fir::BoxAddrOp>(loc, baseAddr);
    retTy = baseAddr.getType();
  }

  mlir::TypeAttr varType = mlir::TypeAttr::get(
      llvm::cast<mlir::omp::PointerLikeType>(retTy).getElementType());

  // For types with unknown extents such as <2x?xi32> we discard the incomplete
  // type info and only retain the base type. The correct dimensions are later
  // recovered through the bounds info.
  if (auto seqType = llvm::dyn_cast<fir::SequenceType>(varType.getValue()))
    if (seqType.hasDynamicExtents())
      varType = mlir::TypeAttr::get(seqType.getEleTy());

  mlir::omp::MapInfoOp op =
      builder.create<mlir::omp::MapInfoOp>(loc, retTy, baseAddr, varType,
          builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
          builder.getAttr<mlir::omp::VariableCaptureKindAttr>(mapCaptureType),
          varPtrPtr, members, membersIndex, bounds, mapperId,
          builder.getStringAttr(name), builder.getBoolAttr(partialMap));
  return op;
}
} // namespace Fortran::common::openmp
