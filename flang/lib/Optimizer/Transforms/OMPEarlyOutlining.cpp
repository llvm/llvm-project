#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"

namespace fir {
#define GEN_PASS_DEF_OMPEARLYOUTLININGPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class OMPEarlyOutliningPass
    : public fir::impl::OMPEarlyOutliningPassBase<OMPEarlyOutliningPass> {

  std::string getOutlinedFnName(llvm::StringRef parentName, unsigned count) {
    return std::string(parentName) + "_omp_outline_" + std::to_string(count);
  }

  // Given a value this function will iterate over an operators results
  // and return the relevant index for the result the value corresponds to.
  // There may be a simpler way to do this however.
  static unsigned getResultIndex(mlir::Value value, mlir::Operation *op) {
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      if (op->getResult(i) == value)
        return i;
    }
    return 0;
  }

  static bool isAddressOfGlobalDeclareTarget(mlir::Value value) {
    if (fir::AddrOfOp addressOfOp =
            mlir::dyn_cast_if_present<fir::AddrOfOp>(value.getDefiningOp()))
      if (fir::GlobalOp gOp = mlir::dyn_cast_if_present<fir::GlobalOp>(
              addressOfOp->getParentOfType<mlir::ModuleOp>().lookupSymbol(
                  addressOfOp.getSymbol())))
        if (auto declareTargetGlobal =
                llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
                    gOp.getOperation()))
          if (declareTargetGlobal.isDeclareTarget())
            return true;
    return false;
  }

  // Currently used for cloning arguments that are nested. Should be
  // extendable where required, perhaps via operation
  // specialisation/overloading, if something needs specialised handling.
  // NOTE: Results in duplication of some values that would otherwise be
  // a single SSA value shared between operations, this is tidied up on
  // lowering to some extent.
  static mlir::Operation *
  cloneArgAndChildren(mlir::OpBuilder &builder, mlir::Operation *op,
                      llvm::SetVector<mlir::Value> &inputs,
                      mlir::Block::BlockArgListType &newInputs) {
    mlir::IRMapping valueMap;
    for (mlir::Value opValue : op->getOperands()) {
      if (opValue.getDefiningOp()) {
        unsigned resIdx = getResultIndex(opValue, opValue.getDefiningOp());
        valueMap.map(opValue,
                     cloneArgAndChildren(builder, opValue.getDefiningOp(),
                                         inputs, newInputs)
                         ->getResult(resIdx));
      } else {
        for (auto inArg : llvm::zip(inputs, newInputs)) {
          if (opValue == std::get<0>(inArg))
            valueMap.map(opValue, std::get<1>(inArg));
        }
      }
    }

    return builder.clone(*op, valueMap);
  }

  static void cloneMapOpVariables(mlir::OpBuilder &builder,
                                  mlir::IRMapping &valueMap,
                                  mlir::IRMapping &mapInfoMap,
                                  llvm::SetVector<mlir::Value> &inputs,
                                  mlir::Block::BlockArgListType &newInputs,
                                  mlir::Value varPtr) {
    if (fir::BoxAddrOp boxAddrOp =
            mlir::dyn_cast_if_present<fir::BoxAddrOp>(varPtr.getDefiningOp())) {
      mlir::Value newV =
          cloneArgAndChildren(builder, boxAddrOp, inputs, newInputs)
              ->getResult(0);
      mapInfoMap.map(varPtr, newV);
      valueMap.map(boxAddrOp, newV);
      return;
    }

    if (isAddressOfGlobalDeclareTarget(varPtr)) {
      fir::AddrOfOp addrOp =
          mlir::dyn_cast<fir::AddrOfOp>(varPtr.getDefiningOp());
      mlir::Value newV = builder.clone(*addrOp)->getResult(0);
      mapInfoMap.map(varPtr, newV);
      valueMap.map(addrOp, newV);
      return;
    }

    for (auto inArg : llvm::zip(inputs, newInputs)) {
      if (varPtr == std::get<0>(inArg))
        mapInfoMap.map(varPtr, std::get<1>(inArg));
    }
  }

  mlir::func::FuncOp outlineTargetOp(mlir::OpBuilder &builder,
                                     mlir::omp::TargetOp &targetOp,
                                     mlir::func::FuncOp &parentFunc,
                                     unsigned count) {
    // NOTE: once implicit captures are handled appropriately in the initial
    // PFT lowering if it is possible, we can remove the usage of
    // getUsedValuesDefinedAbove and instead just iterate over the target op's
    // operands (or just the map arguments) and perhaps refactor this function
    // a little.
    // Collect inputs
    llvm::SetVector<mlir::Value> inputs;
    mlir::Region &targetRegion = targetOp.getRegion();
    mlir::getUsedValuesDefinedAbove(targetRegion, inputs);
    
    // filter out declareTarget and map entries which are specially handled
    // at the moment, so we do not wish these to end up as function arguments
    // which would just be more noise in the IR.
    for (llvm::SetVector<mlir::Value>::iterator iter = inputs.begin(); iter != inputs.end();) {
      if (mlir::isa_and_nonnull<mlir::omp::MapInfoOp>(iter->getDefiningOp()) ||
          isAddressOfGlobalDeclareTarget(*iter)) {
        iter = inputs.erase(iter);
      } else {
        ++iter;
      }
    }

    // Create new function and initialize
    mlir::FunctionType funcType = builder.getFunctionType(
        mlir::TypeRange(inputs.getArrayRef()), mlir::TypeRange());
    std::string parentName(parentFunc.getName());
    std::string funcName = getOutlinedFnName(parentName, count);
    mlir::Location loc = targetOp.getLoc();
    mlir::func::FuncOp newFunc =
        mlir::func::FuncOp::create(loc, funcName, funcType);
    mlir::Block *entryBlock = newFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    mlir::Block::BlockArgListType newInputs = entryBlock->getArguments();

    // Set the declare target information, the outlined function
    // is always a host function.
    if (auto parentDTOp = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
            parentFunc.getOperation()))
      if (auto newDTOp = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
              newFunc.getOperation()))
        newDTOp.setDeclareTarget(mlir::omp::DeclareTargetDeviceType::host,
                                 parentDTOp.getDeclareTargetCaptureClause());

    // Set the early outlining interface parent name
    if (auto earlyOutlineOp =
            llvm::dyn_cast<mlir::omp::EarlyOutliningInterface>(
                newFunc.getOperation()))
      earlyOutlineOp.setParentName(parentName);

    // The value map for the newly generated Target Operation, we must
    // remap most of the input.
    mlir::IRMapping valueMap;

    // Special handling for map, declare target and regular map variables
    // are handled slightly differently for the moment, declare target has
    // its addressOfOp cloned over, whereas we skip it for the regular map
    // variables. We need knowledge of which global is linked to the map
    // operation for declare target, whereas we aren't bothered for the
    // regular map variables for the moment. We could treat both the same,
    // however, cloning across the minimum for the moment to avoid
    // optimisations breaking segments of the lowering seems prudent as this
    // was the original intent of the pass.
    for (mlir::Value oper : targetOp->getOperands()) {
      if (auto mapEntry =
              mlir::dyn_cast<mlir::omp::MapInfoOp>(oper.getDefiningOp())) {
        mlir::IRMapping mapInfoMap;
        for (mlir::Value bound : mapEntry.getBounds()) {
          if (auto mapEntryBound = mlir::dyn_cast<mlir::omp::DataBoundsOp>(
                  bound.getDefiningOp())) {
            mapInfoMap.map(bound, cloneArgAndChildren(builder, mapEntryBound,
                                                      inputs, newInputs)
                                      ->getResult(0));
          }
        }

        cloneMapOpVariables(builder, valueMap, mapInfoMap, inputs, newInputs,
                            mapEntry.getVarPtr());

        if (mapEntry.getVarPtrPtr())
          cloneMapOpVariables(builder, valueMap, mapInfoMap, inputs, newInputs,
                              mapEntry.getVarPtrPtr());

        valueMap.map(
            mapEntry,
            builder.clone(*mapEntry.getOperation(), mapInfoMap)->getResult(0));
      }
    }

    for (auto inArg : llvm::zip(inputs, newInputs))
      valueMap.map(std::get<0>(inArg), std::get<1>(inArg));

    // Clone the target op into the new function
    builder.clone(*(targetOp.getOperation()), valueMap);

    // Create return op
    builder.create<mlir::func::ReturnOp>(loc);

    return newFunc;
  }

  // Returns true if a target region was found int the function.
  bool outlineTargetOps(mlir::OpBuilder &builder,
                        mlir::func::FuncOp &functionOp,
                        mlir::ModuleOp &moduleOp,
                        llvm::SmallVectorImpl<mlir::func::FuncOp> &newFuncs) {
    unsigned count = 0;
    for (auto TargetOp : functionOp.getOps<mlir::omp::TargetOp>()) {
      mlir::func::FuncOp outlinedFunc =
          outlineTargetOp(builder, TargetOp, functionOp, count);
      newFuncs.push_back(outlinedFunc);
      count++;
    }
    return count > 0;
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *context = &getContext();
    mlir::OpBuilder builder(context);
    llvm::SmallVector<mlir::func::FuncOp> newFuncs;

    for (auto functionOp :
         llvm::make_early_inc_range(moduleOp.getOps<mlir::func::FuncOp>())) {
      bool outlined = outlineTargetOps(builder, functionOp, moduleOp, newFuncs);
      if (outlined)
        functionOp.erase();
    }

    for (auto newFunc : newFuncs)
      moduleOp.push_back(newFunc);
  }
};

} // namespace

namespace fir {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOMPEarlyOutliningPass() {
  return std::make_unique<OMPEarlyOutliningPass>();
}
} // namespace fir
