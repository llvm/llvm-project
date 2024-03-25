! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

integer :: i
! CHECK: not yet implemented: Unhandled clause FIRSTPRIVATE in TARGET construct
!$omp target firstprivate(i)
!$omp end target

end program
