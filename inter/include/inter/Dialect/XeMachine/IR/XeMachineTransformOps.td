// XeMachine Transform dialect extension operations.

#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINETRANSFORMOPS_TD
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINETRANSFORMOPS_TD

include "inter/Dialect/XeMachine/IR/XeMachineBase.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def XeMachine_TransformRegAllocLoopOp
    : Op<XeMachineDialect, "transform.regalloc_loop", [
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let cppNamespace = "::inter::xemachine";
  let summary = "run the XeMachine regalloc transform-loop body";
  let arguments = (ins TransformHandleTypeInterface:$target,
                         SymbolRefAttr:$body,
                         DefaultValuedAttr<I64Attr, "32">:$max_iterations);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = [{
    `from` $target `body` `=` $body attr-dict
        `:` functional-type($target, $result)
  }];
}

class XeMachine_RegAllocTransformStage<string mnemonic, string summaryText>
    : Op<XeMachineDialect, mnemonic, [
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let cppNamespace = "::inter::xemachine";
  let summary = summaryText;
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = [{
    `from` $target attr-dict `:` functional-type($target, $result)
  }];
}

def XeMachine_TransformRegAllocBuildStateOp
    : XeMachine_RegAllocTransformStage<"transform.regalloc_build_state",
                                       "build XeMachine regalloc state">;
def XeMachine_TransformRegAllocArfBuildStateOp
    : XeMachine_RegAllocTransformStage<"transform.regalloc_arf_build_state",
                                       "build XeMachine ARF regalloc state">;
def XeMachine_TransformRegAllocArfLinearScanOp
    : XeMachine_RegAllocTransformStage<"transform.regalloc_arf_linear_scan",
                                       "run XeMachine ARF linear scan">;
def XeMachine_TransformRegAllocLinearScanOp
    : XeMachine_RegAllocTransformStage<"transform.regalloc_linear_scan",
                                       "run XeMachine regalloc linear scan">;
def XeMachine_TransformRegAllocRematReliefOp
    : XeMachine_RegAllocTransformStage<"transform.regalloc_remat_relief",
                                       "relieve pressure by rematerialization">;
def XeMachine_TransformRegAllocScratchReliefOp
    : XeMachine_RegAllocTransformStage<"transform.regalloc_scratch_relief",
                                       "relieve pressure through scratch">;

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINETRANSFORMOPS_TD
