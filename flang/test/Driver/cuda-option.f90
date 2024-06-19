! Test -fcuda option
! RUN: %flang_fc1 -cpp -x cuda -fdebug-unparse %s -o - | FileCheck %s
! RUN: not %flang_fc1 -cpp %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR
program main
#if _CUDA
  integer :: var = _CUDA
#endif
  integer, device :: dvar
end program

! CHECK-LABEL: PROGRAM main
! CHECK: INTEGER :: var = 1
! CHECK: INTEGER, DEVICE :: dvar

! ERROR: cuda-option.f90:8:19: error: expected end of statement
