! Test the MLIR pass pipeline

! RUN: bbc --mlir-pass-statistics --mlir-pass-statistics-display=pipeline %s 2>&1 | FileCheck %s

! REQUIRES: asserts

end program

! CHECK: Pass statistics report

! CHECK: Fortran::lower::VerifierPass
! CHECK-NEXT: CSE
! Ideally, we need an output with only the pass names, but
! there is currently no way to get that, so in order to
! guarantee that the passes are in the expected order
! (i.e. use -NEXT) we have to check the statistics output as well.
! CHECK-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
! CHECK-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd

! CHECK-NEXT: 'func.func' Pipeline
! CHECK-NEXT:   ArrayValueCopy
! CHECK-NEXT:   CharacterConversion

! CHECK-NEXT: Canonicalizer
! CHECK-NEXT: SimplifyRegionLite
! CHECK-NEXT: AlgebraicSimplification
! CHECK-NEXT: CSE
! CHECK-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
! CHECK-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd

! CHECK-NEXT: 'func.func' Pipeline
! CHECK-NEXT:   MemoryAllocationOpt

! CHECK-NEXT: Inliner
! CHECK-NEXT: SimplifyRegionLite
! CHECK-NEXT: CSE
! CHECK-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
! CHECK-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd

! CHECK-NEXT: 'func.func' Pipeline
! CHECK-NEXT:   CFGConversion

! CHECK-NEXT: SCFToControlFlow
! CHECK-NEXT: Canonicalizer
! CHECK-NEXT: SimplifyRegionLite
! CHECK-NEXT: CSE
! CHECK-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
! CHECK-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd
! CHECK-NOT: LLVMIRLoweringPass
