! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -mmlir --enable-delayed-privatization=false -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: LINEAR clause without delayed privatization
subroutine linear_no_delayed_privatization()
  integer :: i, x
  !$omp simd linear(x)
  do i = 1, 10
    x = x + 1
  end do
end subroutine
