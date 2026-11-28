! RUN: not %flang_fc1 -fopenmp -fopenmp-version=60 -fsyntax-only %s 2>&1 | FileCheck %s
! Check the coindexing diagnostic independently of other argument restrictions.

subroutine allocate_item
  integer, save :: x[*]
  ! CHECK: A coindexed object is not allowed as a list item on ALLOCATE directive
  !$omp allocate(x[1])
end

subroutine threadprivate_item
  integer, save :: x[*]
  ! CHECK: A coindexed object is not allowed as a list item on THREADPRIVATE directive
  !$omp threadprivate(x[1])
end

subroutine declare_target_item
  integer, save :: x[*]
  ! CHECK: A coindexed object is not allowed as a list item on DECLARE_TARGET directive
  !$omp declare target(x[1])
end

subroutine groupprivate_item
  integer, save :: x[*]
  ! CHECK: A coindexed object is not allowed as a list item on GROUPPRIVATE directive
  !$omp groupprivate(x[1])
end
