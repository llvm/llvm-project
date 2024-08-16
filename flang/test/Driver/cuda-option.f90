! Test -fcuda option
! RUN: %flang_fc1 -cpp -x cuda -fdebug-unparse %s -o - | FileCheck %s
! RUN: not %flang -cpp -x cuda %s -o - 2>&1 | FileCheck %s --check-prefix=MLIRERROR
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

! ERROR: cuda-option.f90:{{.*}}:{{.*}}: error: expected end of statement

! The whole pipeline is not in place yet. It will currently fail at MLIR
! translation level.
! MLIRERROR: failed to legalize operation 'cuf.alloc'
