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
// PASSES-NEXT: Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
// PASSES-NEXT: 'fir.global' Pipeline
// PASSES-NEXT:   SimplifyHLFIRIntrinsics
// PASSES-NEXT:   InlineElementals
// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:   SimplifyHLFIRIntrinsics
// PASSES-NEXT:   InlineElementals
// PASSES-NEXT: 'omp.declare_reduction' Pipeline
// PASSES-NEXT:   SimplifyHLFIRIntrinsics
// PASSES-NEXT:   InlineElementals
// PASSES-NEXT: 'omp.private' Pipeline
// PASSES-NEXT:   SimplifyHLFIRIntrinsics
// PASSES-NEXT:   InlineElementals
// PASSES-NEXT:   Canonicalizer
// PASSES-NEXT:   CSE
// PASSES-NEXT:    (S) 0 num-cse'd - Number of operations CSE'd
// PASSES-NEXT:    (S) 0 num-dce'd - Number of operations DCE'd
// PASSES-NEXT: Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
// PASSES-NEXT: 'fir.global' Pipeline
// PASSES-NEXT:    SimplifyHLFIRIntrinsics
// PASSES-NEXT:    PropagateFortranVariableAttributes
// PASSES-NEXT:    OptimizedBufferization
// PASSES-NEXT:    InlineHLFIRAssign
// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:    SimplifyHLFIRIntrinsics
// PASSES-NEXT:    PropagateFortranVariableAttributes
// PASSES-NEXT:    OptimizedBufferization
// PASSES-NEXT:    InlineHLFIRAssign
// PASSES-NEXT: 'omp.declare_reduction' Pipeline
// PASSES-NEXT:    SimplifyHLFIRIntrinsics
// PASSES-NEXT:    PropagateFortranVariableAttributes
// PASSES-NEXT:    OptimizedBufferization
// PASSES-NEXT:    InlineHLFIRAssign
// PASSES-NEXT: 'omp.private' Pipeline
// PASSES-NEXT:    SimplifyHLFIRIntrinsics
// PASSES-NEXT:    PropagateFortranVariableAttributes
// PASSES-NEXT:    OptimizedBufferization
// PASSES-NEXT:    InlineHLFIRAssign
// PASSES-NEXT:   LowerHLFIROrderedAssignments
// PASSES-NEXT:   LowerHLFIRIntrinsics
// PASSES-NEXT:   BufferizeHLFIR
// PASSES-NEXT: Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
// PASSES-NEXT:   'fir.global' Pipeline
// PASSES-NEXT:     InlineHLFIRAssign
// PASSES-NEXT:   'func.func' Pipeline
// PASSES-NEXT:     InlineHLFIRAssign
// PASSES-NEXT:   'omp.declare_reduction' Pipeline
// PASSES-NEXT:     InlineHLFIRAssign
// PASSES-NEXT:   'omp.private' Pipeline
// PASSES-NEXT:     InlineHLFIRAssign
// PASSES-NEXT:   ConvertHLFIRtoFIR
// PASSES-NEXT:   LowerWorkshare
// PASSES-NEXT:   CSE
// PASSES-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
// PASSES-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd

// PASSES-NEXT: Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
// PASSES-NEXT: 'fir.global' Pipeline
// PASSES-NEXT:   CharacterConversion
// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:   ArrayValueCopy
// PASSES-NEXT:   CharacterConversion
// PASSES-NEXT: 'omp.declare_reduction' Pipeline
// PASSES-NEXT:   CharacterConversion
// PASSES-NEXT: 'omp.private' Pipeline
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

// PASSES-NEXT: PolymorphicOpConversion
// PASSES-NEXT: AssumedRankOpConversion
// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:   OptimizeArrayRepacking
// PASSES-NEXT: LowerRepackArraysPass
// PASSES-NEXT: SimplifyFIROperations

// PASSES-NEXT: Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
// PASSES-NEXT: 'fir.global' Pipeline
// PASSES-NEXT:   StackReclaim
// PASSES-NEXT:   CFGConversion
// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:   StackReclaim
// PASSES-NEXT:   CFGConversion
// PASSES-NEXT: 'omp.declare_reduction' Pipeline
// PASSES-NEXT:   StackReclaim
// PASSES-NEXT:   CFGConversion
// PASSES-NEXT: 'omp.private' Pipeline
// PASSES-NEXT:   StackReclaim
// PASSES-NEXT:   CFGConversion

// PASSES-NEXT: SCFToControlFlow
// PASSES-NEXT: Canonicalizer
// PASSES-NEXT: SimplifyRegionLite
// PASSES-NEXT: CSE
// PASSES-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
// PASSES-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd
// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:   SetRuntimeCallAttributes
// PASSES-NEXT: BoxedProcedurePass
// PASSES-NEXT: AddAliasTags

// PASSES-NEXT: Pipeline Collection : ['fir.global', 'func.func', 'gpu.module', 'omp.declare_reduction', 'omp.private']
// PASSES-NEXT:  'fir.global' Pipeline
// PASSES-NEXT:    AbstractResultOpt
// PASSES-NEXT:  'func.func' Pipeline
// PASSES-NEXT:    AbstractResultOpt
// PASSES-NEXT:  'gpu.module' Pipeline
// PASSES-NEXT:   Pipeline Collection : ['func.func', 'gpu.func'] 
// PASSES-NEXT:   'func.func' Pipeline 
// PASSES-NEXT:   AbstractResultOpt
// PASSES-NEXT:   'gpu.func' Pipeline 
// PASSES-NEXT:   AbstractResultOpt
// PASSES-NEXT:  'omp.declare_reduction' Pipeline
// PASSES-NEXT:    AbstractResultOpt
// PASSES-NEXT:  'omp.private' Pipeline
// PASSES-NEXT:    AbstractResultOpt

// PASSES-NEXT: CodeGenRewrite
// PASSES-NEXT:   (S) 0 num-dce'd - Number of operations eliminated
// PASSES-NEXT: TargetRewrite
// PASSES-NEXT: CompilerGeneratedNamesConversion
// PASSES-NEXT: 'func.func' Pipeline
// PASSES-NEXT:  FunctionAttr
// PASSES-NEXT:  LowerNontemporalPass
// PASSES-NEXT: FIRToLLVMLowering
// PASSES-NEXT: ReconcileUnrealizedCasts
// PASSES-NEXT: LLVMIRLoweringPass
