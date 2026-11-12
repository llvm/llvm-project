
!----------
! RUN lines
!----------
! Try to embed missing file
! RUN: not %flang_fc1 -emit-llvm -o - -fembed-offload-object=%S/Inputs/missing.f90 %s 2>&1 | FileCheck %s --check-prefix=ERROR

! ERROR: error: could not open

parameter(i=1)
integer :: j
end program
