! Test -fcuda option
! RUN: %flang -fc1 -cpp -fcuda -fdebug-unparse %s -o - | FileCheck %s

program main
#if _CUDA
  integer :: var = _CUDA
#endif
  integer, device :: dvar
end program

! CHECK-LABEL: PROGRAM main
! CHECK: INTEGER :: var = 1
! CHECK: INTEGER, DEVICE :: dvar
