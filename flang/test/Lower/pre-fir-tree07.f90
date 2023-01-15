! RUN: %flang_fc1 -fdebug-pre-fir-tree -fopenacc %s | FileCheck %s

! Test structure of the Pre-FIR tree with OpenACC declarative construct

! CHECK: Module m: module m
module m
  real, dimension(10) :: x
  ! CHECK-NEXT: OpenACCDeclarativeConstruct
  !$acc declare create(x)
end
! CHECK: End Module m
