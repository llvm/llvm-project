! RUN: not %flang_fc1 -fopenmp -fopenmp-version=50 -fsyntax-only %s 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -fopenmp -fopenmp-version=51 -fsyntax-only %s 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -fopenmp -fopenmp-version=60 -fsyntax-only %s 2>&1 | FileCheck %s

! [5.2:181], [6.0:315] The allocator argument of USES_ALLOCATORS is an
! expression of allocator_handle type, which omp_lib declares as c_intptr_t.
! The rule holds whether or not the source uses omp_lib, so none of these
! subroutines imports it. The expected kind is matched as a number rather than
! spelled out, because it is the target's C_INTPTR_T kind.

! CHECK: error: The allocator 'wrong_kind_alloc' in a USES_ALLOCATORS clause must be of type INTEGER(KIND={{[0-9]+}}), i.e. OMP_ALLOCATOR_HANDLE_KIND
subroutine uses_allocators_wrong_handle_kind
  ! Deliberately a kind that cannot be C_INTPTR_T on any supported target.
  integer(kind=2) :: wrong_kind_alloc
  integer :: x
  !$omp target uses_allocators(wrong_kind_alloc)
  x = 1
  !$omp end target
end subroutine

! A REAL allocator is diagnosed for its type; the kind rule must not pile a
! second diagnostic onto the same allocator.
! CHECK: error: Must have INTEGER type, but is REAL(4)
! CHECK-NOT: 'not_an_integer' in a USES_ALLOCATORS clause must be of type
subroutine uses_allocators_real_allocator
  real :: not_an_integer
  integer :: x
  !$omp target uses_allocators(not_an_integer)
  x = 1
  !$omp end target
end subroutine
