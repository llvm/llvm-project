// RUN: tco %s | FileCheck %s
// RUN: tco %s --mlir-pass-statistics --mlir-pass-statistics-display=pipeline 2>&1 | FileCheck %s --check-prefix=PASSES

// REQUIRES: asserts

// Check that tco is working with a basic test.
// Also check the passes in the default pipeline.

func.func @_QQmain() {
  return
}

// CHECK: ; ModuleID = 'FIRModule'
// CHECK-LABEL: define void @_QQmain()
// CHECK:       ret void

// PASSES: Pass statistics report

// PASSES:        Canonicalizer
// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:   SimplifyHLFIRIntrinsics
// PASSES-NEXT:   InlineElementals
// PASSES-NEXT:   Canonicalizer
// PASSES-NEXT:   CSE
// PASSES-NEXT:    (S) 0 num-cse'd - Number of operations CSE'd
// PASSES-NEXT:    (S) 0 num-dce'd - Number of operations DCE'd
// PASSES-NEXT:   'func.func' Pipeline
// PASSES-NEXT:    OptimizedBufferization
// PASSES-NEXT:   LowerHLFIROrderedAssignments
// PASSES-NEXT:   LowerHLFIRIntrinsics
// PASSES-NEXT:   BufferizeHLFIR
// PASSES-NEXT:   ConvertHLFIRtoFIR
// PASSES-NEXT:   CSE
// PASSES-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
// PASSES-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd

// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:   ArrayValueCopy
// PASSES-NEXT:   CharacterConversion

// PASSES-NEXT: Canonicalizer
// PASSES-NEXT: SimplifyRegionLite
// PASSES-NEXT: SimplifyIntrinsics
// PASSES-NEXT: AlgebraicSimplification
// PASSES-NEXT: CSE
// PASSES-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
// PASSES-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd

// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:   MemoryAllocationOpt

// PASSES-NEXT: Inliner
// PASSES-NEXT: SimplifyRegionLite
// PASSES-NEXT: CSE
// PASSES-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
// PASSES-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd

// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:   PolymorphicOpConversion
// PASSES-NEXT:   CFGConversion

// PASSES-NEXT: SCFToControlFlow
// PASSES-NEXT: Canonicalizer
// PASSES-NEXT: SimplifyRegionLite
// PASSES-NEXT: CSE
// PASSES-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
// PASSES-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd
// PASSES-NEXT: BoxedProcedurePass

// PASSES-NEXT: Pipeline Collection : ['fir.global', 'func.func']
// PASSES-NEXT:   'fir.global' Pipeline
// PASSES-NEXT:    AbstractResultOnGlobalOpt
// PASSES-NEXT:  'func.func' Pipeline
// PASSES-NEXT:    AbstractResultOnFuncOpt

// PASSES-NEXT: CodeGenRewrite
// PASSES-NEXT:   (S) 0 num-dce'd - Number of operations eliminated
// PASSES-NEXT: TargetRewrite
// PASSES-NEXT: FIRToLLVMLowering
// PASSES-NEXT: LLVMIRLoweringPass
