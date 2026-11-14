! RUN: not %flang_fc1 -fopenmp -fopenmp-version=50 -fsyntax-only %s 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -fopenmp -fopenmp-version=51 -fsyntax-only %s 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s 2>&1 | FileCheck %s

! Before OpenMP 6.0, [5.2:182] asks whether the allocator *is* a predefined
! allocator, so predefined treatment belongs to the entity of the intrinsic
! omp_lib module. A user-defined module named omp_lib does not confer it, even
! when its named constant has the right name and the right handle kind.
!
! This module shadows the intrinsic omp_lib for the whole file, so it is kept
! in a test of its own.

! CHECK: error: A non-predefined allocator 'omp_const_mem_alloc' in a USES_ALLOCATORS clause must be a variable

module omp_lib
  use iso_c_binding, only: c_intptr_t
  integer, parameter :: omp_allocator_handle_kind = c_intptr_t
  integer(omp_allocator_handle_kind), parameter :: omp_const_mem_alloc = 4242
end module

subroutine uses_allocators_user_omp_lib
  use omp_lib
  integer :: x

  !$omp target uses_allocators(omp_const_mem_alloc)
  x = 1
  !$omp end target
end subroutine
