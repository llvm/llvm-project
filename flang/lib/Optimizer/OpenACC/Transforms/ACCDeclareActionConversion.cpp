//===- ACCDeclareActionConversion.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the allocation and deallocation semantics for allocatables and
// pointers in declare directives. OpenACC 3.4, Section 2.13.2: in Fortran, if
// a variable in the declare var-list has the allocatable or pointer attribute,
// then for a non-shared memory device, an allocate (or intrinsic assignment
// that allocates) allocates in both local and device memory and sets the
// dynamic reference counter to one; a deallocate (or assignment that
// deallocates) deallocates from both and sets the counter to zero.
//
// How this pass works:
// - Lowering generates recipe functions that hold the recipe for creating the
//   device copy (using acc dialect operations, e.g. acc.create).
// - Lowering also attaches an attribute to the operations that allocate or
//   deallocate the object.
// - This pass finds operations with that attribute and inserts calls to the
//   corresponding recipe.
//
// Example:
//   module mm
//     real, allocatable :: arr(:)
//     !$acc declare create(arr)
//   contains
//     subroutine sub()
//       allocate(arr(100))
//     end subroutine sub
//   end module mm
//
// Relevant IR before this pass (recipe function and store with attribute):
//   func.func private @_QMmmEarr_acc_declare_update_desc_post_alloc(...) {
//     ...  // acc ops to create/register device copy
//     return
//   }
//   func.func @_QMmmPsub() {
//     ...
//     fir.store %box to %desc {acc.declare_action = #acc.declare_action<
//       postAlloc = @_QMmmEarr_acc_declare_update_desc_post_alloc>} ...
//   }
//
// After this pass (call to recipe inserted after the store):
//   func.func @_QMmmPsub() {
//     ...
//     fir.store %box to %desc ...
//     fir.call @_QMmmEarr_acc_declare_update_desc_post_alloc()
//   }
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/OpenACC/Passes.h"
#include "flang/Optimizer/OpenACC/Support/FIROpenACCUtils.h"
#include "flang/Optimizer/Support/LazySymbolTable.h"
#include "flang/Runtime/entry-names.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "acc-declare-action-conversion"

namespace fir {
namespace acc {
#define GEN_PASS_DEF_ACCDECLAREACTIONCONVERSION
#include "flang/Optimizer/OpenACC/Passes.h.inc"
} // namespace acc
} // namespace fir

using namespace mlir;

namespace {

// Fortran runtime symbol names for pointer allocate/deallocate.
static constexpr llvm::StringRef pointerAllocateName =
    RTNAME_STRING(PointerAllocate);
static constexpr llvm::StringRef pointerDeallocateName =
    RTNAME_STRING(PointerDeallocate);

class ACCDeclareActionConversion
    : public fir::acc::impl::ACCDeclareActionConversionBase<
          ACCDeclareActionConversion> {
public:
  using fir::acc::impl::ACCDeclareActionConversionBase<
      ACCDeclareActionConversion>::ACCDeclareActionConversionBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod);
    fir::LazySymbolTable symbolTable(mod);

    mod.walk([&](Operation *op) {
      auto declareAction = op->getAttrOfType<acc::DeclareActionAttr>(
          acc::getDeclareActionAttrName());
      if (!declareAction)
        return;

      LLVM_DEBUG(llvm::dbgs() << "Found " << acc::getDeclareActionAttrName()
                              << " on: " << *op << "\n");

      auto preAlloc = declareAction.getPreAlloc();
      auto postAlloc = declareAction.getPostAlloc();
      auto preDealloc = declareAction.getPreDealloc();
      auto postDealloc = declareAction.getPostDealloc();

      if (!preAlloc && !postAlloc && !preDealloc && !postDealloc)
        return;

      for (auto action : {preAlloc, postAlloc, preDealloc, postDealloc}) {
        if (!action)
          continue;

        if (auto func = dyn_cast<SymbolRefAttr>(action)) {
          Operation *funcDef = symbolTable.lookupSymbol(func);
          if (!funcDef)
            continue;

          if (auto funcOp = dyn_cast<func::FuncOp>(funcDef))
            if (!funcOp->hasAttr(mlir::acc::getDeclareActionAttrName()))
              funcOp->setAttr(mlir::acc::getDeclareActionAttrName(),
                              mlir::UnitAttr::get(funcOp.getContext()));

          if (action == declareAction.getPreAlloc() ||
              action == declareAction.getPreDealloc())
            builder.setInsertionPoint(op);
          else
            builder.setInsertionPointAfter(op);

          auto funcOp = dyn_cast<func::FuncOp>(funcDef);
          if (!funcOp) {
            op->emitError("declare action callee is not a func.func operation");
            return;
          }
          SmallVector<Value> argVec;
          if (funcOp.getNumArguments() > 0) {
            Value varRef =
                llvm::TypeSwitch<Operation *, Value>(op)
                    .Case<fir::StoreOp>(
                        [&](auto store) { return store.getMemref(); })
                    .Case<fir::BoxAddrOp>(
                        [&](auto boxAddr) { return boxAddr.getVal(); })
                    .Case<fir::CallOp>([&](fir::CallOp call) -> Value {
                      if (auto callee = call.getCalleeAttr()) {
                        StringRef funcName =
                            callee.getLeafReference().getValue();
                        if (funcName == pointerAllocateName ||
                            funcName == pointerDeallocateName) {
                          auto args = call.getArgs();
                          if (args.empty())
                            return {};
                          Value boxRef = args[0];
                          if (!fir::isBoxAddress(boxRef.getType()))
                            return {};
                          return boxRef;
                        }
                      }
                      return {};
                    })
                    .Default([](Operation *) { return Value(); });

            if (!varRef) {
              op->emitError(
                  "could not find argument for declare action recipe call");
              return;
            }
            if (fir::isa_box_type(varRef.getType())) {
              auto loadOp = varRef.getDefiningOp<fir::LoadOp>();
              if (!loadOp) {
                op->emitError("varRef for declare action is not from fir.load");
                return;
              }
              varRef = loadOp.getMemref();
            }
            varRef = fir::acc::getOriginalDef(varRef, /*stripDeclare=*/false);
            // Runtime calls (e.g. PointerAllocate) use ref<box<none>>; recipe
            // expects typed box ref. Look through one convert to get the typed
            // ref when getOriginalDef stopped at the convert (original LRO
            // semantics: do not look through when result is box none).
            Type recipeArgTy = funcOp.getFunctionType().getInput(0);
            if (varRef.getType() != recipeArgTy) {
              if (auto convertOp = varRef.getDefiningOp<fir::ConvertOp>()) {
                Value converted = convertOp.getValue();
                if (converted.getType() == recipeArgTy)
                  varRef = converted;
              }
            }
            if (varRef.getType() != recipeArgTy) {
              op->emitError("declare action recipe expects typed box ref");
              return;
            }
            argVec.push_back(varRef);
          }
          fir::CallOp::create(builder, op->getLoc(), funcOp, argVec);
        }
      }
    });
  }
};

} // namespace
