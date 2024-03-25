! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Do Concurrent in Worksharing loop construct
subroutine sb()
  !$omp do
  do concurrent(i=1:10)
    print *, i
  end do
end subroutine
