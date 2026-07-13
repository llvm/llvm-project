! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Compound clauses of OpenMP ATOMIC COMPARE
program p
  integer :: x
  integer :: r
  integer :: d
  integer :: v
  !$omp atomic compare fail(relaxed)
  if (x .eq. 0) then
     x = 2
  end if
  !$omp end atomic

  !$omp atomic compare capture
  v = x
  if (x > r) then
     x = d
  end if
  !$omp end atomic

  !$omp atomic compare fail(relaxed)
  if (x > r) then
     x = d
  end if
  !$omp end atomic

end program p
