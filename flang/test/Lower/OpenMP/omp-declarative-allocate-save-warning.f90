! Verify that declarative ALLOCATE on SAVE variables or named COMMON blocks
! emits a lowering warning and does not generate omp.allocate_dir /
! omp.allocate_free for those variables.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - 2>&1 | FileCheck %s

subroutine save_allocate_warning
  use omp_lib
  implicit none
  integer, save :: counter = 100

  !$omp allocate(counter) allocator(omp_default_mem_alloc)
end subroutine save_allocate_warning

subroutine common_allocate_warning
  use omp_lib
  implicit none
  real :: cb_a, cb_b
  common /myblock/ cb_a, cb_b

  !$omp allocate(/myblock/) allocator(omp_default_mem_alloc)
end subroutine common_allocate_warning

! Warnings are emitted during lowering before HLFIR is printed.
! CHECK: warning: {{.*}}TODO : OpenMP declarative ALLOCATE on SAVE variables or COMMON blocks is not yet supported, ignoring the ALLOCATE directive for 'counter'
! CHECK: warning: {{.*}}TODO : OpenMP declarative ALLOCATE on SAVE variables or COMMON blocks is not yet supported, ignoring the ALLOCATE directive for 'myblock'

! CHECK-LABEL: func.func @_QPsave_allocate_warning
! CHECK-NOT: omp.allocate_dir
! CHECK-NOT: omp.allocate_free

! CHECK-LABEL: func.func @_QPcommon_allocate_warning
! CHECK-NOT: omp.allocate_dir
! CHECK-NOT: omp.allocate_free
