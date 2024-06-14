! Test the MLIR pass pipeline

! RUN: %flang_fc1 -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline -o /dev/null %s 2>&1 | FileCheck --check-prefixes=ALL %s
! -O0 is the default:
! RUN: %flang_fc1 -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline %s -O0 -o /dev/null 2>&1 | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline %s -O2 -o /dev/null 2>&1 | FileCheck --check-prefixes=ALL,O2 %s

! REQUIRES: asserts

end program

! ALL: Pass statistics report

! ALL: Fortran::lower::VerifierPass
! O2-NEXT: Canonicalizer
! ALL:     Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
! ALL-NEXT:'fir.global' Pipeline
! O2-NEXT:   SimplifyHLFIRIntrinsics
! ALL:       InlineElementals
! ALL-NEXT:'func.func' Pipeline
! O2-NEXT:   SimplifyHLFIRIntrinsics
! ALL:       InlineElementals
! ALL-NEXT:'omp.declare_reduction' Pipeline
! O2-NEXT:   SimplifyHLFIRIntrinsics
! ALL:       InlineElementals
! ALL-NEXT:'omp.private' Pipeline
! O2-NEXT:   SimplifyHLFIRIntrinsics
! ALL:       InlineElementals
! O2-NEXT: Canonicalizer
! O2-NEXT: CSE
! O2-NEXT: (S) {{.*}} num-cse'd
! O2-NEXT: (S) {{.*}} num-dce'd
! O2-NEXT: Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
! O2-NEXT: 'fir.global' Pipeline
! O2-NEXT:   OptimizedBufferization
! O2-NEXT: 'func.func' Pipeline
! O2-NEXT:   OptimizedBufferization
! O2-NEXT: 'omp.declare_reduction' Pipeline
! O2-NEXT:   OptimizedBufferization
! O2-NEXT: 'omp.private' Pipeline
! O2-NEXT:   OptimizedBufferization
! ALL: LowerHLFIROrderedAssignments
! ALL-NEXT: LowerHLFIRIntrinsics
! ALL-NEXT: BufferizeHLFIR
! ALL-NEXT: ConvertHLFIRtoFIR
! ALL-NEXT: CSE
! Ideally, we need an output with only the pass names, but
! there is currently no way to get that, so in order to
! guarantee that the passes are in the expected order
! (i.e. use -NEXT) we have to check the statistics output as well.
! ALL-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
! ALL-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd

! ALL-NEXT: Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
! ALL-NEXT: 'fir.global' Pipeline
! ALL-NEXT:   CharacterConversion
! ALL-NEXT: 'func.func' Pipeline
! ALL-NEXT:   ArrayValueCopy
! ALL-NEXT:   CharacterConversion
! ALL-NEXT: 'omp.declare_reduction' Pipeline
! ALL-NEXT:   CharacterConversion
! ALL-NEXT: 'omp.private' Pipeline
! ALL-NEXT:   CharacterConversion

! ALL-NEXT: Canonicalizer
! ALL-NEXT: SimplifyRegionLite
!  O2-NEXT: SimplifyIntrinsics
!  O2-NEXT: AlgebraicSimplification
! ALL-NEXT: CSE
! ALL-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
! ALL-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd

! ALL-NEXT: 'func.func' Pipeline
! ALL-NEXT:   MemoryAllocationOpt

! ALL-NEXT: Inliner
! ALL-NEXT: SimplifyRegionLite
! ALL-NEXT: CSE
! ALL-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
! ALL-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd

! ALL-NEXT: PolymorphicOpConversion
! ALL-NEXT: AssumedRankOpConversion
! O2-NEXT:  AddAliasTags

! ALL-NEXT: Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
! ALL-NEXT:    'fir.global' Pipeline
! ALL-NEXT:      CFGConversion
! ALL-NEXT:    'func.func' Pipeline
! ALL-NEXT:      CFGConversion
! ALL-NEXT:   'omp.declare_reduction' Pipeline
! ALL-NEXT:      CFGConversion
! ALL-NEXT:   'omp.private' Pipeline
! ALL-NEXT:      CFGConversion

! ALL-NEXT: SCFToControlFlow
! ALL-NEXT: Canonicalizer
! ALL-NEXT: SimplifyRegionLite
! ALL-NEXT: CSE
! ALL-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
! ALL-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd
! ALL-NEXT: BoxedProcedurePass

! ALL-NEXT: Pipeline Collection : ['fir.global', 'func.func', 'omp.declare_reduction', 'omp.private']
! ALL-NEXT:   'fir.global' Pipeline
! ALL-NEXT:    AbstractResultOpt
! ALL-NEXT:  'func.func' Pipeline
! ALL-NEXT:    AbstractResultOpt
! ALL-NEXT:  'omp.declare_reduction' Pipeline
! ALL-NEXT:    AbstractResultOpt
! ALL-NEXT:  'omp.private' Pipeline
! ALL-NEXT:    AbstractResultOpt

! ALL-NEXT: CodeGenRewrite
! ALL-NEXT:   (S) 0 num-dce'd - Number of operations eliminated
! ALL-NEXT: TargetRewrite
! ALL-NEXT: ExternalNameConversion
! ALL-NEXT: FIRToLLVMLowering
! ALL-NOT: LLVMIRLoweringPass
