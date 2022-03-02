// RUN: tco %s | FileCheck %s
// RUN: tco %s --mlir-pass-statistics --mlir-pass-statistics-display=pipeline 2>&1 | FileCheck %s --check-prefix=PASSES

// Check that tco is working with a basic test.
// Also check the passes in the default pipeline.

func @_QQmain() {
  return
}

// CHECK: ; ModuleID = 'FIRModule'
// CHECK-LABEL: define void @_QQmain()
// CHECK:       ret void

// PASSES: Pass statistics report

// PASSES-LABEL: 'func.func' Pipeline
// PASSES: ArrayValueCopy
// PASSES: CharacterConversion
// PASSES: Canonicalizer
// PASSES: SimplifyRegionLite

// PASSES-LABEL: 'func.func' Pipeline
// PASSES: MemoryAllocationOpt
// PASSES: Inliner
// PASSES: CSE

// PASSES-LABEL: 'func.func' Pipeline
// PASSES: CFGConversion
// PASSES: SCFToControlFlow
// PASSES: Canonicalizer
// PASSES: SimplifyRegionLite
// PASSES: BoxedProcedurePass

// PASSES-LABEL: 'func.func' Pipeline
// PASSES: AbstractResultOpt
// PASSES: CodeGenRewrite
// PASSES: TargetRewrite
// PASSES: FIRToLLVMLowering
// PASSES: LLVMIRLoweringPass
