//===-- lib/Support/OpenMP-utils.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Support/OpenMP-utils.h"

#include "aiir/IR/OpDefinition.h"

namespace Fortran::common::openmp {
aiir::Block *genEntryBlock(aiir::OpBuilder &builder, const EntryBlockArgs &args,
    aiir::Region &region) {
  assert(args.isValid() && "invalid args");
  assert(region.empty() && "non-empty region");

  llvm::SmallVector<aiir::Type> types;
  llvm::SmallVector<aiir::Location> locs;
  unsigned numVars = args.hasDeviceAddr.vars.size() + args.hostEvalVars.size() +
      args.inReduction.vars.size() + args.map.vars.size() +
      args.priv.vars.size() + args.reduction.vars.size() +
      args.taskReduction.vars.size() + args.useDeviceAddr.vars.size() +
      args.useDevicePtr.vars.size();
  types.reserve(numVars);
  locs.reserve(numVars);

  auto extractTypeLoc = [&types, &locs](llvm::ArrayRef<aiir::Value> vals) {
    llvm::transform(vals, std::back_inserter(types),
        [](aiir::Value v) { return v.getType(); });
    llvm::transform(vals, std::back_inserter(locs),
        [](aiir::Value v) { return v.getLoc(); });
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
} // namespace Fortran::common::openmp
