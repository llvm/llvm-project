!RUN: not %flang_fc1 -fopenmp -emit-hlfir -o - %s

! Check that we reject the "task" reduction modifier on the "simd" directive.

subroutine fred(x)
  integer, intent(inout) :: x

  !$omp simd reduction(task, +:x)
  do i = 1, 100
    x = foo(i)
  enddo
  !$omp end simd
end
