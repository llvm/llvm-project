//===-- lib/Support/OpenMP-utils.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Support/OpenMP-utils.h"

#include "mlir/IR/OpDefinition.h"

namespace Fortran::common::openmp {
mlir::Block *genEntryBlock(mlir::OpBuilder &builder, const EntryBlockArgs &args,
    mlir::Region &region) {
  assert(region.empty() && "non-empty region");

  llvm::SmallVector<mlir::Type> types;
  llvm::SmallVector<mlir::Location> locs;
  unsigned numVars = args.hasDeviceAddrVars.size() + args.hostEvalVars.size() +
      args.inReductionVars.size() + args.mapVars.size() + args.privVars.size() +
      args.reductionVars.size() + args.taskReductionVars.size() +
      args.useDeviceAddrVars.size() + args.useDevicePtrVars.size();
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
  extractTypeLoc(args.hasDeviceAddrVars);
  extractTypeLoc(args.hostEvalVars);
  extractTypeLoc(args.inReductionVars);
  extractTypeLoc(args.mapVars);
  extractTypeLoc(args.privVars);
  extractTypeLoc(args.reductionVars);
  extractTypeLoc(args.taskReductionVars);
  extractTypeLoc(args.useDeviceAddrVars);
  extractTypeLoc(args.useDevicePtrVars);

  return builder.createBlock(&region, {}, types, locs);
}
} // namespace Fortran::common::openmp
