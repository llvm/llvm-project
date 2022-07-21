// RUN: tco %s | FileCheck %s
// RUN: tco %s --mlir-pass-statistics --mlir-pass-statistics-display=pipeline 2>&1 | FileCheck %s --check-prefix=PASSES

// Check that tco is working with a basic test.
// Also check the passes in the default pipeline.

func.func @_QQmain() {
  return
}

// CHECK: ; ModuleID = 'FIRModule'
// CHECK-LABEL: define void @_QQmain()
// CHECK:       ret void

// PASSES: Pass statistics report

// PASSES:      CSE
// PASSES-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
// PASSES-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd

// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:   ArrayValueCopy
// PASSES-NEXT:   CharacterConversion

// PASSES-NEXT: Canonicalizer
// PASSES-NEXT: SimplifyRegionLite
// PASSES-NEXT: AlgebraicSimplification
// PASSES-NEXT: CSE
// PASSES-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
// PASSES-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd

// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:   MemoryAllocationOpt

// PASSES-NEXT: Inliner
// PASSES-NEXT: CSE
// PASSES-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
// PASSES-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd

// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:   CFGConversion

// PASSES-NEXT: SCFToControlFlow
// PASSES-NEXT: Canonicalizer
// PASSES-NEXT: SimplifyRegionLite
// PASSES-NEXT: CSE
// PASSES-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
// PASSES-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd
// PASSES-NEXT: BoxedProcedurePass

// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:   AbstractResultOpt

// PASSES-NEXT: CodeGenRewrite
// PASSES-NEXT:   (S) 0 num-dce'd - Number of operations eliminated
// PASSES-NEXT: TargetRewrite
// PASSES-NEXT: FIRToLLVMLowering
// PASSES-NEXT: LLVMIRLoweringPass
