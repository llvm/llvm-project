! Test the debug pass pipeline

! RUN: %flang -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline -o /dev/null %s 2>&1 | FileCheck --check-prefixes=ALL,NO-DEBUG %s

! RUN: %flang -g0 -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline %s -o /dev/null 2>&1 | FileCheck --check-prefixes=ALL,NO-DEBUG %s
! RUN: %flang -g -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline %s -o /dev/null 2>&1 | FileCheck --check-prefixes=ALL,DEBUG %s
! RUN: %flang -g1 -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline %s -o /dev/null 2>&1 | FileCheck --check-prefixes=ALL,DEBUG %s
! RUN: %flang -gline-tables-only -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline %s -o /dev/null 2>&1 | FileCheck --check-prefixes=ALL,DEBUG %s
! RUN: %flang -gline-directives-only -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline %s -o /dev/null 2>&1 | FileCheck --check-prefixes=ALL,DEBUG,DEBUG-DIRECTIVES %s
! RUN: %flang -g2 -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline %s -o /dev/null 2>&1 | FileCheck --check-prefixes=ALL,DEBUG,DEBUG-CONSTRUCT %s
! RUN: %flang -g3 -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline %s -o /dev/null 2>&1 | FileCheck --check-prefixes=ALL,DEBUG,DEBUG-CONSTRUCT %s

! RUN: not %flang_fc1 -debug-info-kind=invalid -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline %s -o /dev/null 2>&1 | FileCheck --check-prefixes=DEBUG-ERR %s

! REQUIRES: asserts

end program

! DEBUG-CONSTRUCT: warning: Unsupported debug option: constructor
! DEBUG-DIRECTIVES: warning: Unsupported debug option: line-directives-only
!
! DEBUG-ERR: error: invalid value 'invalid' in '-debug-info-kind=invalid'
! DEBUG-ERR-NOT: Pass statistics report

! ALL: Pass statistics report

! ALL: Fortran::lower::VerifierPass
! ALL-NEXT: 'func.func' Pipeline
! ALL-NEXT:   InlineElementals
! ALL-NEXT: LowerHLFIROrderedAssignments
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

! ALL-NEXT: 'func.func' Pipeline
! ALL-NEXT:   ArrayValueCopy
! ALL-NEXT:   CharacterConversion

! ALL-NEXT: Canonicalizer
! ALL-NEXT: SimplifyRegionLite
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

! ALL-NEXT: Pipeline Collection : ['func.func', 'omp.declare_reduction']
! ALL-NEXT:   'func.func' Pipeline
! ALL-NEXT:     PolymorphicOpConversion
! ALL-NEXT:     CFGConversionOnFunc
! ALL-NEXT:   'omp.declare_reduction' Pipeline
! ALL-NEXT:     CFGConversionOnReduction
! ALL-NEXT: SCFToControlFlow
! ALL-NEXT: Canonicalizer
! ALL-NEXT: SimplifyRegionLite
! ALL-NEXT: CSE
! ALL-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
! ALL-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd
! ALL-NEXT: BoxedProcedurePass

! ALL-NEXT: Pipeline Collection : ['fir.global', 'func.func']
! ALL-NEXT:   'fir.global' Pipeline
! ALL-NEXT:   AbstractResultOnGlobalOpt
! ALL-NEXT:   'func.func' Pipeline
! ALL-NEXT:   AbstractResultOnFuncOpt

! ALL-NEXT: CodeGenRewrite
! ALL-NEXT:   (S) 0 num-dce'd - Number of operations eliminated
! ALL-NEXT: TargetRewrite
! ALL-NEXT: ExternalNameConversion
! DEBUG-NEXT: AddDebugFoundation
! NO-DEBUG-NOT: AddDebugFoundation
! ALL: FIRToLLVMLowering
! ALL-NOT: LLVMIRLoweringPass
