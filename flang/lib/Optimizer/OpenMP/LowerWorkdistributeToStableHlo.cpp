//===- LowerWorkdistribute.cpp
//-------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering and optimisations of omp.workdistribute.
//
// Fortran array statements are lowered to fir as fir.do_loop unordered.
// lower-workdistribute pass works mainly on identifying fir.do_loop unordered
// that is nested in target{teams{workdistribute{fir.do_loop unordered}}} and
// lowers it to target{teams{parallel{distribute{wsloop{loop_nest}}}}}.
// It hoists all the other ops outside target region.
// Relaces heap allocation on target with omp.target_allocmem and
// deallocation with omp.target_freemem from host. Also replaces
// runtime function "Assign" with omp_target_memcpy.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/OpenMP/Utils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/DebugLog.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <optional>
#include <variant>

namespace flangomp {
#define GEN_PASS_DEF_LOWERWORKDISTRIBUTETOJIT
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "lower-workdistribute-to-stablehlo"

using namespace mlir;

namespace {
class LowerWorkdistributeToJitPass
    : public flangomp::impl::LowerWorkdistributeToJitBase<
          LowerWorkdistributeToJitPass> {
public:
  void runOnOperation() override {
    MLIRContext &context = getContext();
    auto moduleOp = getOperation();

    SmallVector<omp::TargetOp> targetOps;
    moduleOp.walk(
        [&](omp::TargetOp targetOp) { targetOps.push_back(targetOp); });
    for (auto targetOp : targetOps) {
      if (targetOp
              ->walk(
                  [](omp::WorkdistributeOp) { return WalkResult::interrupt(); })
              .wasInterrupted() == false) {
        LDBG() << "Ignoring non-workdistribute target op:\n" << *targetOp;
        continue;
      }

      std::string str;
      {
        OpBuilder b(&context);
        b.setInsertionPointToStart(moduleOp.getBody());
        auto f = func::FuncOp::create(
            b, targetOp.getLoc(), "kernel",
            FunctionType::get(&context,
                              targetOp.getRegion().begin()->getArgumentTypes(),
                              {}));
        b.cloneRegionBefore(targetOp.getRegion(), f.getRegion(),
                            f.getRegion().begin());

        llvm::raw_string_ostream os(str);
        os << *f;
        f->erase();
      }

      OpBuilder b(targetOp);
      auto targetJitOp = omp::TargetJitOp::create(
          b, targetOp.getLoc(), targetOp.getAllocateVars(),
          targetOp.getAllocatorVars(), targetOp.getBareAttr(),
          targetOp.getDependKindsAttr(), targetOp.getDependVars(),
          targetOp.getDevice(), targetOp.getHasDeviceAddrVars(),
          targetOp.getHostEvalVars(), targetOp.getIfExpr(),
          targetOp.getInReductionVars(), targetOp.getInReductionByrefAttr(),
          targetOp.getInReductionSymsAttr(), targetOp.getIsDevicePtrVars(),
          targetOp.getMapVars(), targetOp.getNowaitAttr(),
          targetOp.getPrivateVars(), targetOp.getPrivateSymsAttr(),
          targetOp.getPrivateNeedsBarrierAttr(), targetOp.getThreadLimit(),
          targetOp.getPrivateMapsAttr(),
          /*jit_code=*/StringAttr::get(&context, str.c_str()));
      targetJitOp.getRegion().takeBody(targetOp.getRegion());
      targetJitOp.getRegion().begin()->clear();
      b.setInsertionPointToStart(&*targetJitOp.getRegion().begin());
      omp::TerminatorOp::create(b, targetOp->getLoc());
      targetOp->erase();
    }
  }
};
} // namespace
