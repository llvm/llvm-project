! Verify that declarative ALLOCATE on SAVE variables or named COMMON blocks
! emits a lowering warning and does not generate omp.allocate_dir /
! omp.allocate_free for those variables.

! RUN: %flang_fc1 -emit-hlfir %openmp_flags %s -o - 2>&1 | FileCheck %s

subroutine save_allocate_warning
  implicit none
  integer, save :: counter = 100

  !$omp allocate(counter) allocator(1)
end subroutine save_allocate_warning

subroutine implicit_save_allocate_warning
  implicit none
  integer :: implicit_counter = 100

  !$omp allocate(implicit_counter) allocator(1)
end subroutine implicit_save_allocate_warning

subroutine common_allocate_warning
  implicit none
  real :: cb_a, cb_b
  common /myblock/ cb_a, cb_b

  !$omp allocate(/myblock/) allocator(1)
end subroutine common_allocate_warning

! Warnings are emitted during lowering before HLFIR is printed.
! CHECK: warning: {{.*}}TODO : OpenMP declarative ALLOCATE on SAVE variables or COMMON blocks is not yet supported, ignoring the ALLOCATE directive for 'counter'
! CHECK: warning: {{.*}}TODO : OpenMP declarative ALLOCATE on SAVE variables or COMMON blocks is not yet supported, ignoring the ALLOCATE directive for 'implicit_counter'
! CHECK: warning: {{.*}}TODO : OpenMP declarative ALLOCATE on SAVE variables or COMMON blocks is not yet supported, ignoring the ALLOCATE directive for 'myblock'

! CHECK-LABEL: func.func @_QPsave_allocate_warning
! CHECK-NOT: omp.allocate_dir
! CHECK-NOT: omp.allocate_free

! CHECK-LABEL: func.func @_QPimplicit_save_allocate_warning
! CHECK-NOT: omp.allocate_dir
! CHECK-NOT: omp.allocate_free

! CHECK-LABEL: func.func @_QPcommon_allocate_warning
! CHECK-NOT: omp.allocate_dir
! CHECK-NOT: omp.allocate_free
