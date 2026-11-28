! Verify that declarative ALLOCATE on dummy arguments emits a lowering
! warning for OpenMP versions earlier than 6.0 and does not generate
! omp.allocate_dir / omp.allocate_free for those variables.

! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 %s -o - 2>&1 | FileCheck %s

subroutine dummy_allocate_warning(x)
  implicit none
  integer :: x

  !$omp allocate(x) allocator(1)
end subroutine dummy_allocate_warning

! Warnings are emitted during lowering before HLFIR is printed.
! CHECK: warning: {{.*}}TODO : OpenMP declarative ALLOCATE on dummy arguments is not yet supported, ignoring the ALLOCATE directive for 'x'

! CHECK-LABEL: func.func @_QPdummy_allocate_warning
! CHECK-NOT: omp.allocate_dir
! CHECK-NOT: omp.allocate_free
