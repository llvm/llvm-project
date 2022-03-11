! RUN: %flang_fc1 -fdebug-pre-fir-tree -fopenacc %s | FileCheck %s

! Test structure of the Pre-FIR tree with OpenACC declarative construct

! CHECK: ModuleLike
module m
  real, dimension(10) :: x
  ! CHECK-NEXT: OpenACCDeclarativeConstruct
  !$acc declare create(x)
end
! CHECK: End ModuleLike

