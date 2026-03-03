//===- EHABILowering.cpp - Lower flattened CIR EH ops to ABI-specific form ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that lowers ABI-agnostic flattened CIR exception
// handling operations into an ABI-specific form. Currently only the Itanium
// C++ ABI is supported.
//
// The Itanium ABI lowering performs these transformations:
//   - cir.eh.initiate      → cir.eh.inflight_exception (landing pad)
//   - cir.eh.dispatch      → cir.eh.typeid + cir.cmp + cir.brcond chains
//   - cir.begin_cleanup    → (removed)
//   - cir.end_cleanup      → (removed)
//   - cir.begin_catch      → call to __cxa_begin_catch
//   - cir.end_catch        → call to __cxa_end_catch
//   - cir.resume           → cir.resume.flat
//   - !cir.eh_token values → (!cir.ptr<!void>, !u32i) value pairs
//   - personality function set on functions requiring EH
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/Builders.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/TargetParser/Triple.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CIREHABILOWERING
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Shared utilities
//===----------------------------------------------------------------------===//

/// Ensure a function with the given name and type exists in the module. If it
/// does not exist, create a private external declaration.
static cir::FuncOp getOrCreateRuntimeFuncDecl(mlir::ModuleOp mod,
                                              mlir::Location loc,
                                              StringRef name,
                                              cir::FuncType funcTy) {
  if (auto existing = mod.lookupSymbol<cir::FuncOp>(name))
    return existing;

  mlir::OpBuilder builder(mod.getContext());
  builder.setInsertionPointToEnd(mod.getBody());
  auto funcOp = cir::FuncOp::create(builder, loc, name, funcTy);
  funcOp.setLinkageAttr(cir::GlobalLinkageKindAttr::get(
      builder.getContext(), cir::GlobalLinkageKind::ExternalLinkage));
  funcOp.setPrivate();
  return funcOp;
}

//===----------------------------------------------------------------------===//
// EH ABI Lowering Base Class
//===----------------------------------------------------------------------===//

/// Abstract base class for exception-handling ABI lowering.
/// Each supported ABI (Itanium, Microsoft, etc.) provides a concrete subclass.
class EHABILowering {
public:
  explicit EHABILowering(mlir::ModuleOp mod)
      : mod(mod), ctx(mod.getContext()), builder(ctx) {}
  virtual ~EHABILowering() = default;

  /// Lower all EH operations in the module to an ABI-specific form.
  virtual mlir::LogicalResult run() = 0;

protected:
  mlir::ModuleOp mod;
  mlir::MLIRContext *ctx;
  mlir::OpBuilder builder;
};

//===----------------------------------------------------------------------===//
// Itanium EH ABI Lowering
//===----------------------------------------------------------------------===//

/// Lowers flattened CIR EH operations to the Itanium C++ ABI form.
///
/// The entry point is run(), which iterates over all functions and
/// calls lowerFunc() for each. lowerFunc() drives all lowering from
/// cir.eh.initiate operations: every other EH op (begin/end_cleanup,
/// eh.dispatch, begin/end_catch, resume) is reachable by tracing the
/// eh_token produced by the initiate through its users.
class ItaniumEHLowering : public EHABILowering {
public:
  using EHABILowering::EHABILowering;
  mlir::LogicalResult run() override;

private:
  /// Maps a !cir.eh_token value to its Itanium ABI replacement pair:
  /// an exception pointer (!cir.ptr<!void>) and a type id (!u32i).
  using EhTokenMap = DenseMap<mlir::Value, std::pair<mlir::Value, mlir::Value>>;

  cir::VoidType voidType;
  cir::PointerType voidPtrType;
  cir::PointerType u8PtrType;
  cir::IntType u32Type;

  // Cached runtime function declarations, initialized when needed by
  // ensureRuntimeDecls().
  cir::FuncOp personalityFunc;
  cir::FuncOp beginCatchFunc;
  cir::FuncOp endCatchFunc;

  void ensureRuntimeDecls(mlir::Location loc);
  mlir::LogicalResult lowerFunc(cir::FuncOp funcOp);
  void lowerEhInitiate(cir::EhInitiateOp initiateOp, cir::FuncOp funcOp,
                       EhTokenMap &ehTokenMap,
                       SmallVectorImpl<mlir::Operation *> &deadOps);
  void lowerDispatch(cir::EhDispatchOp dispatch, mlir::Value exnPtr,
                     mlir::Value typeId, cir::FuncOp funcOp,
                     SmallVectorImpl<mlir::Operation *> &deadOps);
};

/// Lower all EH operations in the module to the Itanium-specific form.
mlir::LogicalResult ItaniumEHLowering::run() {
  // Pre-compute the common types used throughout all function lowerings.
  // TODO(cir): Move these to the base class if they are also needed for MSVC.
  voidType = cir::VoidType::get(ctx);
  voidPtrType = cir::PointerType::get(voidType);
  auto u8Type = cir::IntType::get(ctx, 8, /*isSigned=*/false);
  u8PtrType = cir::PointerType::get(u8Type);
  u32Type = cir::IntType::get(ctx, 32, /*isSigned=*/false);

  for (cir::FuncOp funcOp : mod.getOps<cir::FuncOp>()) {
    if (mlir::failed(lowerFunc(funcOp)))
      return mlir::failure();
  }
  return mlir::success();
}

/// Ensure the necessary Itanium runtime function declarations exist in the
/// module.
void ItaniumEHLowering::ensureRuntimeDecls(mlir::Location loc) {
  // TODO(cir): Handle other personality functions. This probably isn't needed
  // here if we fix codegen to always set the personality function.
  if (!personalityFunc) {
    auto s32Type = cir::IntType::get(ctx, 32, /*isSigned=*/true);
    auto personalityFuncTy = cir::FuncType::get({}, s32Type, /*isVarArg=*/true);
    personalityFunc = getOrCreateRuntimeFuncDecl(
        mod, loc, "__gxx_personality_v0", personalityFuncTy);
  }

  if (!beginCatchFunc) {
    auto beginCatchFuncTy =
        cir::FuncType::get({voidPtrType}, u8PtrType, /*isVarArg=*/false);
    beginCatchFunc = getOrCreateRuntimeFuncDecl(mod, loc, "__cxa_begin_catch",
                                                beginCatchFuncTy);
  }

  if (!endCatchFunc) {
    auto endCatchFuncTy = cir::FuncType::get({}, voidType, /*isVarArg=*/false);
    endCatchFunc =
        getOrCreateRuntimeFuncDecl(mod, loc, "__cxa_end_catch", endCatchFuncTy);
  }
}

/// Lower all EH operations in a single function.
mlir::LogicalResult ItaniumEHLowering::lowerFunc(cir::FuncOp funcOp) {
  if (funcOp.isDeclaration())
    return mlir::success();

  // All EH lowering follows from cir.eh.initiate operations. The token each
  // initiate produces connects it to every other EH op in the function
  // (begin/end_cleanup, eh.dispatch, begin/end_catch, resume) through the
  // token graph. A single walk to collect initiates is therefore sufficient.
  SmallVector<cir::EhInitiateOp> initiateOps;
  funcOp.walk([&](cir::EhInitiateOp op) { initiateOps.push_back(op); });
  if (initiateOps.empty())
    return mlir::success();

  ensureRuntimeDecls(funcOp.getLoc());

  // Set the personality function if it is not already set.
  // TODO(cir): The personality function should already have been set by this
  // point. If we've seen a try operation, it will have been set by
  // emitCXXTryStmt. If we only have cleanups, it may not have been set. We
  // need to fix that in CodeGen. This is a placeholder until that is done.
  if (!funcOp.getPersonality())
    funcOp.setPersonality("__gxx_personality_v0");

  // Lower each initiate and all EH ops connected to it. The token map is
  // shared across all initiate operations. Multiple initiates may flow into the
  // same dispatch block, and the map ensures the arguments are registered
  // only once. Dispatch ops are scheduled for deferred removal so that sibling
  // initiates can still read catch types from a shared dispatch.
  EhTokenMap ehTokenMap;
  SmallVector<mlir::Operation *> deadOps;
  for (cir::EhInitiateOp initiateOp : initiateOps)
    lowerEhInitiate(initiateOp, funcOp, ehTokenMap, deadOps);

  // Erase operations that were deferred during per-initiate processing
  // (dispatch ops whose catch types were read by multiple initiates).
  for (mlir::Operation *op : deadOps)
    op->erase();

  // Remove the !cir.eh_token block arguments that were replaced by (ptr, u32)
  // pairs. Iterate in reverse to preserve argument indices during removal.
  for (mlir::Block &block : funcOp.getBody()) {
    for (int i = block.getNumArguments() - 1; i >= 0; --i) {
      if (mlir::isa<cir::EhTokenType>(block.getArgument(i).getType()))
        block.eraseArgument(i);
    }
  }

  return mlir::success();
}

/// Lower all EH operations connected to a single cir.eh.initiate.
///
/// The cir.eh.initiate is the root of a token graph. The token it produces
/// flows through branch edges to consuming operations:
///
///   cir.eh.initiate → (via cir.br) → cir.begin_cleanup
///                                     → cir.end_cleanup (via cleanup_token)
///                                   → (via cir.br) → cir.eh.dispatch
///                                                     → (successors) →
///                                                       cir.begin_catch
///                                                       → cir.end_catch
///                                                         (via catch_token)
///                                   → cir.resume
///
/// A single traversal of the token graph discovers and processes every
/// connected op inline. The inflight_exception is created up-front without
/// a catch_type_list; when the dispatch is encountered during traversal,
/// the catch types are read and set on the inflight op.
///
/// Dispatch ops are not erased during per-initiate processing because they may
/// be used by other initiate ops that haven't yet been lowered. Instead they
/// are added to \p deadOps and erased by the caller after all initiates have
/// been lowered.
///
/// \p ehTokenMap is shared across all initiates in the function so that block
/// arguments reachable from multiple sibling initiates are registered once.
void ItaniumEHLowering::lowerEhInitiate(
    cir::EhInitiateOp initiateOp, cir::FuncOp funcOp, EhTokenMap &ehTokenMap,
    SmallVectorImpl<mlir::Operation *> &deadOps) {
  mlir::Value rootToken = initiateOp.getEhToken();

  // Create the inflight_exception without a catch_type_list. The catch types
  // will be set once we encounter the dispatch during the traversal below.
  builder.setInsertionPoint(initiateOp);
  auto inflightOp = cir::EhInflightOp::create(
      builder, initiateOp.getLoc(), /*cleanup=*/initiateOp.getCleanup(),
      /*catch_type_list=*/mlir::ArrayAttr{});

  ehTokenMap[rootToken] = {inflightOp.getExceptionPtr(),
                           inflightOp.getTypeId()};

  // Single traversal of the token graph. For each token value (the root token
  // or a block argument that carries it), we snapshot its users, register
  // (ptr, u32) replacement arguments on successor blocks, then process every
  // user inline. This avoids collecting ops into separate vectors.
  SmallVector<mlir::Value, 4> worklist;
  SmallPtrSet<mlir::Value, 8> visited;
  worklist.push_back(rootToken);

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    // Snapshot users before modifying any of them (erasing ops during
    // iteration would invalidate the use-list iterator).
    SmallVector<mlir::Operation *, 4> users;
    for (mlir::OpOperand &use : current.getUses())
      users.push_back(use.getOwner());

    // Register replacement block arguments on successor blocks (extending the
    // worklist), then lower the op itself.
    for (mlir::Operation *user : users) {
      // Trace into successor blocks to register (ptr, u32) replacement
      // arguments for any !cir.eh_token block arguments found there.  Even
      // if a block arg was already registered by a sibling initiate, it is
      // still added to the worklist so that the traversal can reach the
      // shared dispatch to read catch types.
      for (unsigned s = 0; s < user->getNumSuccessors(); ++s) {
        mlir::Block *succ = user->getSuccessor(s);
        for (mlir::BlockArgument arg : succ->getArguments()) {
          if (!mlir::isa<cir::EhTokenType>(arg.getType()))
            continue;
          if (!ehTokenMap.count(arg)) {
            mlir::Value ptrArg = succ->addArgument(voidPtrType, arg.getLoc());
            mlir::Value u32Arg = succ->addArgument(u32Type, arg.getLoc());
            ehTokenMap[arg] = {ptrArg, u32Arg};
          }
          worklist.push_back(arg);
        }
      }

      if (auto op = mlir::dyn_cast<cir::BeginCleanupOp>(user)) {
        // begin_cleanup / end_cleanup are no-ops for Itanium.  Erase the
        // end_cleanup first (drops the cleanup_token use) then the begin.
        for (auto &tokenUsers :
             llvm::make_early_inc_range(op.getCleanupToken().getUses())) {
          if (auto endOp =
                  mlir::dyn_cast<cir::EndCleanupOp>(tokenUsers.getOwner()))
            endOp.erase();
        }
        op.erase();
      } else if (auto op = mlir::dyn_cast<cir::BeginCatchOp>(user)) {
        // Replace end_catch → __cxa_end_catch (drops the catch_token use),
        // then replace begin_catch → __cxa_begin_catch.
        for (auto &tokenUsers :
             llvm::make_early_inc_range(op.getCatchToken().getUses())) {
          if (auto endOp =
                  mlir::dyn_cast<cir::EndCatchOp>(tokenUsers.getOwner())) {
            builder.setInsertionPoint(endOp);
            cir::CallOp::create(builder, endOp.getLoc(),
                                mlir::FlatSymbolRefAttr::get(endCatchFunc),
                                voidType, mlir::ValueRange{});
            endOp.erase();
          }
        }

        auto [exnPtr, typeId] = ehTokenMap.lookup(op.getEhToken());
        builder.setInsertionPoint(op);
        auto callOp = cir::CallOp::create(
            builder, op.getLoc(), mlir::FlatSymbolRefAttr::get(beginCatchFunc),
            u8PtrType, mlir::ValueRange{exnPtr});
        mlir::Value castResult = callOp.getResult();
        mlir::Type expectedPtrType = op.getExnPtr().getType();
        if (castResult.getType() != expectedPtrType)
          castResult =
              cir::CastOp::create(builder, op.getLoc(), expectedPtrType,
                                  cir::CastKind::bitcast, callOp.getResult());
        op.getExnPtr().replaceAllUsesWith(castResult);
        op.erase();
      } else if (auto op = mlir::dyn_cast<cir::EhDispatchOp>(user)) {
        // Read catch types from the dispatch and set them on the inflight op.
        mlir::ArrayAttr catchTypes = op.getCatchTypesAttr();
        if (catchTypes && catchTypes.size() > 0) {
          SmallVector<mlir::Attribute> typeSymbols;
          for (mlir::Attribute attr : catchTypes)
            typeSymbols.push_back(
                mlir::cast<cir::GlobalViewAttr>(attr).getSymbol());
          inflightOp.setCatchTypeListAttr(builder.getArrayAttr(typeSymbols));
        }
        // Only lower the dispatch once. A sibling initiate sharing the same
        // dispatch will still read its catch types (above), but the comparison
        // chain and branch replacement are only created the first time.
        if (!llvm::is_contained(deadOps, op.getOperation())) {
          auto [exnPtr, typeId] = ehTokenMap.lookup(op.getEhToken());
          lowerDispatch(op, exnPtr, typeId, funcOp, deadOps);
        }
      } else if (auto op = mlir::dyn_cast<cir::ResumeOp>(user)) {
        auto [exnPtr, typeId] = ehTokenMap.lookup(op.getEhToken());
        builder.setInsertionPoint(op);
        cir::ResumeFlatOp::create(builder, op.getLoc(), exnPtr, typeId);
        op.erase();
      } else if (auto op = mlir::dyn_cast<cir::BrOp>(user)) {
        // Replace eh_token operands with the (ptr, u32) pair.
        SmallVector<mlir::Value> newOperands;
        bool changed = false;
        for (mlir::Value operand : op.getDestOperands()) {
          auto it = ehTokenMap.find(operand);
          if (it != ehTokenMap.end()) {
            newOperands.push_back(it->second.first);
            newOperands.push_back(it->second.second);
            changed = true;
          } else {
            newOperands.push_back(operand);
          }
        }
        if (changed) {
          builder.setInsertionPoint(op);
          cir::BrOp::create(builder, op.getLoc(), op.getDest(), newOperands);
          op.erase();
        }
      }
    }
  }

  initiateOp.erase();
}

/// Lower a cir.eh.dispatch by creating a comparison chain in new blocks.
/// The dispatch itself is replaced with a branch to the first comparison
/// block and added to deadOps for deferred removal.
void ItaniumEHLowering::lowerDispatch(
    cir::EhDispatchOp dispatch, mlir::Value exnPtr, mlir::Value typeId,
    cir::FuncOp funcOp, SmallVectorImpl<mlir::Operation *> &deadOps) {
  mlir::Location dispLoc = dispatch.getLoc();
  mlir::Block *defaultDest = dispatch.getDefaultDestination();
  mlir::ArrayAttr catchTypes = dispatch.getCatchTypesAttr();
  mlir::SuccessorRange catchDests = dispatch.getCatchDestinations();
  mlir::Block *dispatchBlock = dispatch->getBlock();

  // Build the comparison chain in new blocks inserted after the dispatch's
  // block. The dispatch itself is replaced with a branch to the first
  // comparison block and scheduled for deferred removal.
  if (!catchTypes || catchTypes.empty()) {
    // No typed catches: replace dispatch with a direct branch.
    builder.setInsertionPoint(dispatch);
    cir::BrOp::create(builder, dispLoc, defaultDest,
                      mlir::ValueRange{exnPtr, typeId});
  } else {
    unsigned numCatches = catchTypes.size();

    // Create and populate comparison blocks in reverse order so that each
    // block's false destination (the next comparison block, or defaultDest
    // for the last one) is already available. Each createBlock inserts
    // before the previous one, so the blocks end up in forward order.
    mlir::Block *insertBefore = dispatchBlock->getNextNode();
    mlir::Block *falseDest = defaultDest;
    mlir::Block *firstCmpBlock = nullptr;
    for (int i = numCatches - 1; i >= 0; --i) {
      auto *cmpBlock = builder.createBlock(insertBefore, {voidPtrType, u32Type},
                                           {dispLoc, dispLoc});

      mlir::Value cmpExnPtr = cmpBlock->getArgument(0);
      mlir::Value cmpTypeId = cmpBlock->getArgument(1);

      auto globalView = mlir::cast<cir::GlobalViewAttr>(catchTypes[i]);
      auto ehTypeIdOp =
          cir::EhTypeIdOp::create(builder, dispLoc, globalView.getSymbol());
      auto cmpOp = cir::CmpOp::create(builder, dispLoc, cir::CmpOpKind::eq,
                                      cmpTypeId, ehTypeIdOp.getTypeId());

      cir::BrCondOp::create(builder, dispLoc, cmpOp, catchDests[i], falseDest,
                            mlir::ValueRange{cmpExnPtr, cmpTypeId},
                            mlir::ValueRange{cmpExnPtr, cmpTypeId});

      insertBefore = cmpBlock;
      falseDest = cmpBlock;
      firstCmpBlock = cmpBlock;
    }

    // Replace the dispatch with a branch to the first comparison block.
    builder.setInsertionPoint(dispatch);
    cir::BrOp::create(builder, dispLoc, firstCmpBlock,
                      mlir::ValueRange{exnPtr, typeId});
  }

  // Schedule the dispatch for deferred removal. We cannot erase it now because
  // a sibling initiate that shares this dispatch may still need to read its
  // catch types.
  deadOps.push_back(dispatch);
}

//===----------------------------------------------------------------------===//
// The Pass
//===----------------------------------------------------------------------===//

struct CIREHABILoweringPass
    : public impl::CIREHABILoweringBase<CIREHABILoweringPass> {
  CIREHABILoweringPass() = default;
  void runOnOperation() override;
};

void CIREHABILoweringPass::runOnOperation() {
  auto mod = mlir::cast<mlir::ModuleOp>(getOperation());

  // The target triple is attached to the module as the "cir.triple" attribute.
  // If it is absent (e.g. a CIR module parsed from text without a triple) we
  // cannot determine the ABI and must skip the pass.
  auto tripleAttr = mlir::dyn_cast_if_present<mlir::StringAttr>(
      mod->getAttr(cir::CIRDialect::getTripleAttrName()));
  if (!tripleAttr) {
    mod.emitError("Module has no target triple");
    return;
  }

  // Select the ABI-specific lowering handler from the triple. The Microsoft
  // C++ ABI targets a Windows MSVC environment; everything else uses Itanium.
  // Extend this when Microsoft ABI lowering is added.
  llvm::Triple triple(tripleAttr.getValue());
  std::unique_ptr<EHABILowering> lowering;
  if (triple.isWindowsMSVCEnvironment()) {
    mod.emitError(
        "EH ABI lowering is not yet implemented for the Microsoft ABI");
    return signalPassFailure();
  } else {
    lowering = std::make_unique<ItaniumEHLowering>(mod);
  }

  if (mlir::failed(lowering->run()))
    return signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::createCIREHABILoweringPass() {
  return std::make_unique<CIREHABILoweringPass>();
}
