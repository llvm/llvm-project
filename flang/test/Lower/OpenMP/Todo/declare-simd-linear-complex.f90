! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: declare simd linear step rescaling for non-integer/non-float types (complex, character, derived)

subroutine foo(x)
  implicit none
  complex, allocatable, intent(inout) :: x
  !$omp declare simd linear(x : ref, step(2))
  x = x + (1.0, 0.0)
end subroutine
