! Test the MLIR pass pipeline

! RUN: %flang_fc1 -flang-experimental-lower-through-core-mlir -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline %s -o /dev/null 2>&1 | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -flang-experimental-lower-through-core-mlir -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline %s -O2 -o /dev/null 2>&1 | FileCheck --check-prefixes=ALL,O2 %s

! REQUIRES: asserts

end program

! ALL: FIRToCoreMLIRPass
! ALL-NEXT: FoldMemRefAliasOpsPass
! O2-NEXT: Mem2Reg
! O2-NEXT:   (S) 0 new block args - Total amount of new block argument inserted in blocks
! O2-NEXT:   (S) 0 promoted slots - Total amount of memory slot promoted
! O2-NEXT: CSE
! O2-NEXT:   (S) 0 num-cse'd - Number of operations CSE'd
! O2-NEXT:   (S) 0 num-dce'd - Number of operations DCE'd
! O2-NEXT: Canonicalizer
! ALL-NEXT: ExpandOpsPass
! ALL-NEXT: ExpandStridedMetadataPass
! ALL-NEXT: LowerAffinePass
! ALL-NEXT: FinalizeMemRefToLLVMConversionPass
! ALL-NEXT: FIRToLLVMLowering
! ALL-NEXT: ReconcileUnrealizedCastsPass
