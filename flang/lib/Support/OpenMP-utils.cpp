//===-- lib/Support/OpenMP-utils.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Support/OpenMP-utils.h"
#include "flang/Optimizer/Builder/DirectivesCommon.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"

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

bool needsBoundsOps(mlir::Value var) {
  assert(mlir::isa<mlir::omp::PointerLikeType>(var.getType()) &&
      "only pointer like types expected");
  mlir::Type t = fir::unwrapRefType(var.getType());
  if (mlir::Type inner = fir::dyn_cast_ptrOrBoxEleTy(t))
    return fir::hasDynamicSize(inner);
  return fir::hasDynamicSize(t);
}

void genBoundsOps(fir::FirOpBuilder &builder, mlir::Value var,
    llvm::SmallVectorImpl<mlir::Value> &boundsOps) {
  mlir::Location loc = var.getLoc();
  fir::factory::AddrAndBoundsInfo info =
      fir::factory::getDataOperandBaseAddr(builder, var,
          /*isOptional=*/false, loc);
  fir::ExtendedValue exv =
      hlfir::translateToExtendedValue(loc, builder, hlfir::Entity{info.addr},
          /*contiguousHint=*/true)
          .first;
  llvm::SmallVector<mlir::Value> tmp =
      fir::factory::genImplicitBoundsOps<mlir::omp::MapBoundsOp,
          mlir::omp::MapBoundsType>(
          builder, info, exv, /*dataExvIsAssumedSize=*/false, loc);
  llvm::append_range(boundsOps, tmp);
}
} // namespace Fortran::common::openmp
