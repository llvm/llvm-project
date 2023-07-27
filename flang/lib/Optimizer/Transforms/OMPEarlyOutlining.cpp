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

  mlir::func::FuncOp outlineTargetOp(mlir::OpBuilder &builder,
                                     mlir::omp::TargetOp &targetOp,
                                     mlir::func::FuncOp &parentFunc,
                                     unsigned count) {
    // Collect inputs
    llvm::SetVector<mlir::Value> inputs;
    for (auto operand : targetOp.getOperation()->getOperands())
      inputs.insert(operand);

    mlir::Region &targetRegion = targetOp.getRegion();
    mlir::getUsedValuesDefinedAbove(targetRegion, inputs);

    // Create new function and initialize
    mlir::FunctionType funcType = builder.getFunctionType(
        mlir::TypeRange(inputs.getArrayRef()), mlir::TypeRange());
    std::string parentName(parentFunc.getName());
    std::string funcName = getOutlinedFnName(parentName, count);
    auto loc = targetOp.getLoc();
    mlir::func::FuncOp newFunc =
        mlir::func::FuncOp::create(loc, funcName, funcType);
    mlir::Block *entryBlock = newFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    mlir::ValueRange newInputs = entryBlock->getArguments();

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

    // Create input map from inputs to function parameters.
    mlir::IRMapping valueMap;
    for (auto InArg : llvm::zip(inputs, newInputs))
      valueMap.map(std::get<0>(InArg), std::get<1>(InArg));

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
