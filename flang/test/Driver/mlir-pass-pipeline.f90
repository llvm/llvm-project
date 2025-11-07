! REQUIRES: asserts
!
! Test the MLIR pass pipeline
!
! The table below summarizes the relationship between the optimization levels
! -O<N> and the integer speedup and size levels.
!
!          .------------------------------.
!          |  Level  |  Speedup  |  Size  |
!          |------------------------------|
!          |   -O0   |     0     |   0    |
!          |   -O1   |     1     |   0    |
!          |   -O2   |     2     |   0    |
!          |   -O3   |     3     |   0    |
!          |   -Os   |     2     |   1    |
!          |   -Oz   |     2     |   2    |
!          '------------------------------'
!
! Since the speedup level for -Os and -Oz is the same as that of -O2, most of
! the passes that are run at -O2 are run at -Os and -Oz as well, except for any
! passes that might increase the size of the code. The names of the FileCheck
! prefixes in the RUN below lines indicate at which optimization levels the
! corresponding output is expected. For instance:
!
!     ALL    O0, O1, O2, O3, Os, Oz (note that -O1 and -O3 are not explicitly
!            tested here)
!     O2     O2 only, but not Os or Oz
!     O02    O0 and O2, but not Os or Oz
!     O2SZ   O2, Os, and Oz
!
! NOTE: At the time of writing, the same set of MLIR passes is run for both
! optimization level -Os and -Oz.
!
! RUN: %flang_fc1 -S -o /dev/null %s -mmlir --mlir-pass-statistics \
! RUN:     -mmlir --mlir-pass-statistics-display=pipeline 2>&1 \
! RUN:     | FileCheck --check-prefixes=ALL,O02 %s
!
! RUN: %flang_fc1 -O0 -S -o /dev/null %s -mmlir --mlir-pass-statistics \
! RUN:     -mmlir --mlir-pass-statistics-display=pipeline 2>&1 \
! RUN:     | FileCheck --check-prefixes=ALL,O02 %s
!
! RUN: %flang_fc1 -O2 -S -o /dev/null %s -mmlir --mlir-pass-statistics \
! RUN:     -mmlir --mlir-pass-statistics-display=pipeline 2>&1 \
! RUN:     | FileCheck --check-prefixes=ALL,O02,O2,O2SZ %s
!
! RUN: %flang_fc1 -Os -S -o /dev/null %s -mmlir --mlir-pass-statistics \
! RUN:     -mmlir --mlir-pass-statistics-display=pipeline 2>&1 \
! RUN:     | FileCheck --check-prefixes=ALL,O2 %s
!
! RUN: %flang_fc1 -Oz -S -o /dev/null %s -mmlir --mlir-pass-statistics \
! RUN:     -mmlir --mlir-pass-statistics-display=pipeline 2>&1 \
! RUN:     | FileCheck --check-prefixes=ALL,O2 %s
!
! Ideally, we need an output with only the pass names, but there is currently no
! way to get that, so in order to guarantee that the passes are in the expected
! order (i.e. use -NEXT) we have to check the statistics output as well.
!
! ALL:         Pass statistics report
! ALL:         Fortran::lower::VerifierPass
! ALL:         Pass statistics report
! ALL:         Fortran::lower::VerifierPass
! O2-NEXT:     Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
! O2-NEXT:       'fir.global' Pipeline
! O2-NEXT:         ExpressionSimplification
! O2-NEXT:       'func.func' Pipeline
! O2-NEXT:         ExpressionSimplification
! O2-NEXT:       'omp.declare_reduction' Pipeline
! O2-NEXT:         ExpressionSimplification
! O2-NEXT:       'omp.private' Pipeline
! O2-NEXT:         ExpressionSimplification
! O2-NEXT:     Canonicalizer
! ALL-NEXT:    Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
! ALL-NEXT:      'fir.global' Pipeline
! O2-NEXT:         SimplifyHLFIRIntrinsics
! O02-NEXT:        InlineElementals
! ALL-NEXT:      'func.func' Pipeline
! O2-NEXT:         SimplifyHLFIRIntrinsics
! O02-NEXT:        InlineElementals
! ALL-NEXT:      'omp.declare_reduction' Pipeline
! O2-NEXT:         SimplifyHLFIRIntrinsics
! O02-NEXT:        InlineElementals
! ALL-NEXT:      'omp.private' Pipeline
! O2-NEXT:         SimplifyHLFIRIntrinsics
! O02-NEXT:        InlineElementals
! O2-NEXT:     Canonicalizer
! O2-NEXT:     CSE
! O2-NEXT:       (S) {{.*}} num-cse'd
! O2-NEXT:       (S) {{.*}} num-dce'd
! O2-NEXT:     Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
! O2-NEXT:       'fir.global' Pipeline
! O2-NEXT:         SimplifyHLFIRIntrinsics
! O2-NEXT:         PropagateFortranVariableAttributes
! O2-NEXT:         OptimizedBufferization
! O2SZ-NEXT:       InlineHLFIRAssign
! O2-NEXT:       'func.func' Pipeline
! O2-NEXT:         SimplifyHLFIRIntrinsics
! O2-NEXT:         PropagateFortranVariableAttributes
! O2-NEXT:         OptimizedBufferization
! O2SZ-NEXT:       InlineHLFIRAssign
! O2-NEXT:       'omp.declare_reduction' Pipeline
! O2-NEXT:         SimplifyHLFIRIntrinsics
! O2-NEXT:         PropagateFortranVariableAttributes
! O2-NEXT:         OptimizedBufferization
! O2SZ-NEXT:       InlineHLFIRAssign
! O2-NEXT:       'omp.private' Pipeline
! O2-NEXT:         SimplifyHLFIRIntrinsics
! O2-NEXT:         PropagateFortranVariableAttributes
! O2-NEXT:         OptimizedBufferization
! O2SZ-NEXT:       InlineHLFIRAssign
! ALL:         LowerHLFIROrderedAssignments
! ALL-NEXT:    LowerHLFIRIntrinsics
! ALL-NEXT:    BufferizeHLFIR
! O2SZ-NEXT:   Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
! O2SZ-NEXT:     'fir.global' Pipeline
! O2SZ-NEXT:       InlineHLFIRAssign
! O2SZ-NEXT:     'func.func' Pipeline
! O2SZ-NEXT:       InlineHLFIRAssign
! O2SZ-NEXT:     'omp.declare_reduction' Pipeline
! O2SZ-NEXT:       InlineHLFIRAssign
! O2SZ-NEXT:     'omp.private' Pipeline
! O2SZ-NEXT:       InlineHLFIRAssign
! ALL-NEXT:    ConvertHLFIRtoFIR
! ALL-NEXT:    CSE
! ALL-NEXT:      (S) 0 num-cse'd - Number of operations CSE'd
! ALL-NEXT:      (S) 0 num-dce'd - Number of operations DCE'd
! ALL-NEXT:    Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
! ALL-NEXT:      'fir.global' Pipeline
! ALL-NEXT:        CharacterConversion
! ALL-NEXT:      'func.func' Pipeline
! ALL-NEXT:        ArrayValueCopy
! ALL-NEXT:        CharacterConversion
! ALL-NEXT:      'omp.declare_reduction' Pipeline
! ALL-NEXT:        CharacterConversion
! ALL-NEXT:      'omp.private' Pipeline
! ALL-NEXT:        CharacterConversion
! ALL-NEXT:    Canonicalizer
! ALL-NEXT:    SimplifyRegionLite
! O2SZ-NEXT:   SimplifyIntrinsics
! O2SZ-NEXT:   AlgebraicSimplification
! ALL-NEXT:    CSE
! ALL-NEXT:      (S) 0 num-cse'd - Number of operations CSE'd
! ALL-NEXT:      (S) 0 num-dce'd - Number of operations DCE'd
! ALL-NEXT:    'func.func' Pipeline
! ALL-NEXT:      MemoryAllocationOpt
! ALL-NEXT:    Inliner
! ALL-NEXT:    SimplifyRegionLite
! ALL-NEXT:    CSE
! ALL-NEXT:      (S) 0 num-cse'd - Number of operations CSE'd
! ALL-NEXT:      (S) 0 num-dce'd - Number of operations DCE'd
! ALL-NEXT:    PolymorphicOpConversion
! ALL-NEXT:    AssumedRankOpConversion
! O2-NEXT:     'func.func' Pipeline
! O2-NEXT:       OptimizeArrayRepacking
! ALL-NEXT:    LowerRepackArraysPass
! ALL-NEXT:    SimplifyFIROperations
! ALL-NEXT:    Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
! ALL-NEXT:      'fir.global' Pipeline
! ALL-NEXT:        StackReclaim
! ALL-NEXT:        CFGConversion
! ALL-NEXT:      'func.func' Pipeline
! ALL-NEXT:        StackReclaim
! ALL-NEXT:        CFGConversion
! ALL-NEXT:      'omp.declare_reduction' Pipeline
! ALL-NEXT:        StackReclaim
! ALL-NEXT:        CFGConversion
! ALL-NEXT:      'omp.private' Pipeline
! ALL-NEXT:        StackReclaim
! ALL-NEXT:        CFGConversion
! ALL-NEXT:    SCFToControlFlow
! ALL-NEXT:    Canonicalizer
! ALL-NEXT:    SimplifyRegionLite
! ALL-NEXT:    ConvertComplexPow
! ALL-NEXT:    CSE
! ALL-NEXT:      (S) 0 num-cse'd - Number of operations CSE'd
! ALL-NEXT:      (S) 0 num-dce'd - Number of operations DCE'd
! O2-NEXT:     'func.func' Pipeline
! O2-NEXT:       SetRuntimeCallAttributes
! ALL-NEXT:    MIFOpConversion
! ALL-NEXT:    BoxedProcedurePass
! O2-NEXT:     AddAliasTags
! ALL-NEXT:   Pipeline Collection : ['fir.global', 'func.func', 'gpu.module', 'omp.declare_reduction', 'omp.private']
! ALL-NEXT:     'fir.global' Pipeline
! ALL-NEXT:        AbstractResultOpt
! ALL-NEXT:     'func.func' Pipeline
! ALL-NEXT:        AbstractResultOpt
! ALL-NEXT:     'gpu.module' Pipeline
! ALL-NEXT:       Pipeline Collection : ['func.func', 'gpu.func']
! ALL-NEXT:         'func.func' Pipeline
! ALL-NEXT:           AbstractResultOpt
! ALL-NEXT:         'gpu.func' Pipeline
! ALL-NEXT:           AbstractResultOpt
! ALL-NEXT:     'omp.declare_reduction' Pipeline
! ALL-NEXT:       AbstractResultOpt
! ALL-NEXT:     'omp.private' Pipeline
! ALL-NEXT:       AbstractResultOpt
! ALL-NEXT:    CodeGenRewrite
! ALL-NEXT:      (S) 0 num-dce'd - Number of operations eliminated
! ALL-NEXT:    ExternalNameConversion
! ALL-NEXT:    TargetRewrite
! ALL-NEXT:    CompilerGeneratedNamesConversion
! ALL-NEXT:    'func.func' Pipeline
! ALL-NEXT:      FunctionAttr
! ALL-NEXT:    FIRToLLVMLowering
! ALL-NOT:     LLVMIRLoweringPass

end program
