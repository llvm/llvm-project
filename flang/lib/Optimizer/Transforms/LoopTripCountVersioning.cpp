//===- LoopTripCountVersioning.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Versions an innermost loop whose runtime trip count is likely to equal a
/// compile-time constant N:
///
///   if (ub == N) { copy of the loop with ub := N } else { original loop }
///
/// Both arms are the original loop with only the bound changed, so the
/// transformation is legal wherever the guard is uniform, which holds on a CPU
/// and not on a GPU: a thread-varying guard could split threads across two
/// copies of a barrier, so device code is excluded. The constant trip count
/// lets LLVM fully unroll the fast arm, or vectorize it without a remainder
/// loop.
///
/// Heuristic: the loop variable is the leading subscript of an explicit-shape
/// array whose declared leading extent N is at most max-section-extent. For
/// `a(1:m, :) = 0` with `real :: a(4, n)`, the inner loop is versioned on
/// `m == 4`. Loops are skipped when N times the number of FIR operations in
/// the body exceeds max-unrolled-ops, a proxy for whether LLVM will fully
/// unroll the fast arm. This runs on FIR, the only level where both the loop
/// and the array's declared type exist.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

namespace fir {
#define GEN_PASS_DEF_LOOPTRIPCOUNTVERSIONING
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-loop-tripcount-versioning"

STATISTIC(NumLoopsExamined, "Number of innermost loops examined");
STATISTIC(NumLoopsTooLarge,
          "Number of candidates skipped: N * body size > max-unrolled-ops");
STATISTIC(NumLoopsVersioned, "Number of loops versioned on trip count");

namespace {

/// One specialized copy of a loop: when \p condition holds at runtime, the
/// loop runs exactly \p tripCount iterations. Both values must dominate the
/// loop.
struct LoopVersion {
  mlir::Value condition;
  mlir::Value tripCount;
};

/// An innermost `do i = 1, ub` whose induction variable is the leading
/// subscript of an array with constant declared leading extent \p extent.
struct SectionExtentCandidate {
  fir::DoLoopOp loop;
  int64_t extent;
};

class LoopTripCountVersioningPass
    : public fir::impl::LoopTripCountVersioningBase<
          LoopTripCountVersioningPass> {
public:
  LoopTripCountVersioningPass() = default;
  LoopTripCountVersioningPass(
      const fir::LoopTripCountVersioningOptions &options)
      : Base(options) {}
  void runOnOperation() override;
};

} // namespace

/// Look through fir.convert operations.
static mlir::Value stripConverts(mlir::Value val) {
  while (auto cvt = val.getDefiningOp<fir::ConvertOp>())
    val = cvt.getValue();
  return val;
}

/// Is \p val the integer constant \p expected, looking through converts?
static bool isConstantInt(mlir::Value val, int64_t expected) {
  std::optional<llvm::APInt> cst = fir::getIntIfConstant(stripConverts(val));
  return cst && cst->trySExtValue() == expected;
}

/// The shape operand is optional on fir.array_coor. An absent shape or a plain
/// fir.shape carries only extents, so the lower bound is one in every
/// dimension; only a fir.shape_shift can say otherwise.
static bool hasUnitLeadingLowerBound(mlir::Value shapeLike) {
  if (!shapeLike || shapeLike.getDefiningOp<fir::ShapeOp>())
    return true;
  auto shapeShift = shapeLike.getDefiningOp<fir::ShapeShiftOp>();
  return shapeShift && isConstantInt(shapeShift.getOrigins().front(), 1);
}

/// If the leading subscript of \p coor is \p iv, and the array's leading
/// dimension has a constant declared extent onto which \p iv maps one-to-one
/// (unit lower bound and, if sliced, a triplet `(1, sectionUb, 1)`), return
/// the extent.
///
/// Only the leading dimension is considered: it becomes the innermost loop,
/// and unrolling produces contiguous memory accesses. An array section that
/// reaches the loop through a temporary box (fir.box_addr of a fir.embox) is
/// not recognized. Both are possible future extensions.
static std::optional<int64_t> getLeadingExtentIndexedBy(fir::ArrayCoorOp coor,
                                                        mlir::Value iv,
                                                        mlir::Value sectionUb) {
  // unwrapRefType leaves a box type as is, so this also rejects boxed arrays.
  auto seqTy = mlir::dyn_cast<fir::SequenceType>(
      fir::unwrapRefType(coor.getMemref().getType()));
  if (!seqTy || seqTy.getShape().empty())
    return std::nullopt;
  int64_t extent = seqTy.getShape().front();
  if (extent == fir::SequenceType::getUnknownExtent())
    return std::nullopt;
  if (!hasUnitLeadingLowerBound(coor.getShape()))
    return std::nullopt;

  mlir::ValueRange indices = coor.getIndices();
  if (indices.empty() || stripConverts(indices.front()) != iv)
    return std::nullopt;
  if (mlir::Value slice = coor.getSlice()) {
    auto sliceOp = slice.getDefiningOp<fir::SliceOp>();
    if (!sliceOp)
      return std::nullopt;
    mlir::ValueRange triples = sliceOp.getTriples();
    if (!isConstantInt(triples[0], 1) || !isConstantInt(triples[2], 1) ||
        stripConverts(triples[1]) != sectionUb)
      return std::nullopt;
  }
  return extent;
}

/// If \p loop is an innermost `do i = 1, ub step 1` with a non-constant upper
/// bound whose induction variable is the leading subscript of at least one
/// array with a constant declared leading extent, return the smallest such
/// extent N, provided 1 < N <= \p maxExtent. The smallest extent is the value
/// ub must equal when every such array is at full extent, which is the case
/// shape conformance forces for array-section assignments.
///
/// Accesses under a fir.if count too: a WHERE mask lowers to a fir.if, and
/// `b` in `where (m(1:n)) a(1:n) = b` forces n == size(b) whether or not it
/// is touched. TODO: a user conditional like `if (c) b(i) = 0` looks the
/// same and does not bound ub; distinguishing the two needs a mark from
/// HLFIR lowering. Counting it only picks a smaller N than is likely.
///
/// Extent 1 is skipped: legacy code declares dummies as `a(1)` and indexes
/// past them, so `ub == 1` would almost never hold.
static std::optional<int64_t> getSectionExtent(fir::DoLoopOp loop,
                                               unsigned maxExtent) {
  // A constant upper bound needs no versioning.
  if (!isConstantInt(loop.getLowerBound(), 1) ||
      !isConstantInt(loop.getStep(), 1) ||
      fir::getIntIfConstant(stripConverts(loop.getUpperBound())))
    return std::nullopt;

  // Array-section lowering clamps the loop bound to max(ub, 0) while the slice
  // triplet carries the unclamped ub; compare against the unclamped value.
  mlir::Value ub = loop.getUpperBound();
  mlir::Value sectionUb = ub;
  if (auto maxOp = ub.getDefiningOp<mlir::arith::MaxSIOp>()) {
    if (isConstantInt(maxOp.getRhs(), 0))
      sectionUb = maxOp.getLhs();
    else if (isConstantInt(maxOp.getLhs(), 0))
      sectionUb = maxOp.getRhs();
  }
  sectionUb = stripConverts(sectionUb);

  mlir::Value iv = loop.getInductionVar();
  std::optional<int64_t> extent;
  loop.getBody()->walk([&](fir::ArrayCoorOp coor) {
    if (std::optional<int64_t> coorExtent =
            getLeadingExtentIndexedBy(coor, iv, sectionUb))
      extent = extent ? std::min(*extent, *coorExtent) : *coorExtent;
  });
  if (!extent || *extent <= 1 || *extent > maxExtent)
    return std::nullopt;
  return extent;
}

/// Replace \p loop with a chain of guarded copies, one per entry of
/// \p versions, each with its upper bound replaced by the constant trip
/// count, and the original loop as the final else arm:
///
///   if (c0) { loop with ub := tc0 }
///   else if (c1) { loop with ub := tc1 }
///   else { original loop }
///
/// The conditions and trip counts must dominate \p loop. Loop results are
/// forwarded through fir.result.
static void versionLoop(mlir::OpBuilder &builder, fir::DoLoopOp loop,
                        llvm::ArrayRef<LoopVersion> versions) {
  assert(!versions.empty() && "nothing to version on");
  mlir::Location loc = loop.getLoc();
  mlir::TypeRange resultTypes = loop->getResultTypes();
  bool hasResults = !resultTypes.empty();

  builder.setInsertionPoint(loop);
  fir::IfOp outerIf;
  for (const LoopVersion &version : versions) {
    auto ifOp = fir::IfOp::create(builder, loc, resultTypes, version.condition,
                                  /*withElse=*/true);
    if (!outerIf) {
      outerIf = ifOp;
    } else if (hasResults) {
      // Terminate the enclosing else arm with this nested if's results.
      fir::ResultOp::create(builder, loc, ifOp.getResults());
    }

    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    auto fastLoop = mlir::cast<fir::DoLoopOp>(builder.clone(*loop));
    fastLoop.getUpperBoundMutable().assign(version.tripCount);
    if (hasResults)
      fir::ResultOp::create(builder, loc, fastLoop.getResults());

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  }

  // The insertion point is now at the start of the innermost else arm. Move
  // the original loop there.
  loop->replaceAllUsesWith(outerIf);
  loop->remove();
  builder.insert(loop);
  if (hasResults)
    fir::ResultOp::create(builder, loc, loop.getResults());
}

/// Device code may contain convergent operations which must not be duplicated
/// under a condition that can differ between threads. Three things make a loop
/// device code: a device ancestor (cuf.kernel, gpu.launch, acc compute region,
/// omp.target), a device function (cuf.proc_attr, acc routine, omp
/// declare_target nohost), or a device module (omp.is_target_device, where
/// every remaining function is compiled for the target). attributes(host,
/// device) and unspecialized acc routines count: their device copy is split
/// off after this pass runs, so the one func.func here would be versioned for
/// both.
static bool isNoHostDeclareTarget(mlir::func::FuncOp func) {
  auto iface = mlir::dyn_cast<mlir::omp::DeclareTargetInterface>(*func);
  if (!iface)
    return false;
  mlir::omp::DeclareTargetAttr attr = iface.getDeclareTarget();
  return attr &&
         attr.getDeviceType() == mlir::omp::DeclareTargetDeviceType::nohost;
}

static bool runsOnDevice(fir::DoLoopOp loop) {
  if (cuf::isExecutingOnDevice(loop) ||
      loop->getParentOfType<mlir::omp::TargetOp>())
    return true;
  auto func = loop->getParentOfType<mlir::func::FuncOp>();
  if (func && (cuf::isDeviceProcedure(func) || mlir::acc::isAccRoutine(func) ||
               isNoHostDeclareTarget(func)))
    return true;
  auto offloadMod = loop->getParentOfType<mlir::omp::OffloadModuleInterface>();
  return offloadMod && offloadMod.getIsTargetDevice();
}

/// Innermost means no nested loop of any kind. Loops containing OpenMP or
/// OpenACC operations are also excluded.
static bool isVersionableInnermostLoop(fir::DoLoopOp loop) {
  return !loop.getBody()
              ->walk([](mlir::Operation *op) {
                if (mlir::isa<mlir::LoopLikeOpInterface>(op) ||
                    mlir::isa_and_present<mlir::omp::OpenMPDialect,
                                          mlir::acc::OpenACCDialect>(
                        op->getDialect()))
                  return mlir::WalkResult::interrupt();
                return mlir::WalkResult::advance();
              })
              .wasInterrupted();
}

void LoopTripCountVersioningPass::runOnOperation() {
  if (maxSectionExtent == 0)
    return;
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");
  mlir::func::FuncOp func = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Func-name:" << func.getSymName() << "\n");

  // Collect candidates before transforming anything. Candidates are innermost
  // loops, so no candidate contains another and the transformations are
  // independent.
  llvm::SmallVector<SectionExtentCandidate> candidates;
  func.walk([&](fir::DoLoopOp loop) {
    if (runsOnDevice(loop) || !isVersionableInnermostLoop(loop))
      return;
    ++NumLoopsExamined;
    std::optional<int64_t> extent = getSectionExtent(loop, maxSectionExtent);
    if (!extent)
      return;
    unsigned bodyOps = 0;
    loop.getBody()->walk([&](mlir::Operation *) { ++bodyOps; });
    uint64_t unrolledOps = static_cast<uint64_t>(*extent) * bodyOps;
    bool tooLarge = maxUnrolledOps != 0 && unrolledOps > maxUnrolledOps;
    LLVM_DEBUG({
      // One write per line so parallel builds do not interleave mid-line.
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "Section extent candidate (N=" << *extent
         << ", body ops=" << bodyOps << ") at " << loop.getLoc()
         << (tooLarge ? " skipped: too large to unroll\n" : "\n");
      llvm::dbgs() << msg;
    });
    if (tooLarge) {
      ++NumLoopsTooLarge;
      return;
    }
    candidates.push_back({loop, *extent});
  });

  if (candidates.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "No loops to transform.\n=== End " DEBUG_TYPE " ===\n");
    return;
  }

  mlir::OpBuilder builder(func.getContext());

  for (const SectionExtentCandidate &candidate : candidates) {
    fir::DoLoopOp loop = candidate.loop;
    mlir::Location loc = loop.getLoc();
    mlir::Value ub = loop.getUpperBound();

    // Define the guard values before the loop so they dominate the fir.if.
    builder.setInsertionPoint(loop);
    mlir::Value extentCst = mlir::arith::ConstantOp::create(
        builder, loc, ub.getType(),
        builder.getIntegerAttr(ub.getType(), candidate.extent));
    mlir::Value isDeclaredExtent = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, ub, extentCst);

    versionLoop(builder, loop, {{isDeclaredExtent, extentCst}});
    ++NumLoopsVersioned;
  }

  LLVM_DEBUG(llvm::dbgs() << "Func after transformation:\n");
  LLVM_DEBUG(func->dump());
  LLVM_DEBUG(llvm::dbgs() << "=== End " DEBUG_TYPE " ===\n");
}
