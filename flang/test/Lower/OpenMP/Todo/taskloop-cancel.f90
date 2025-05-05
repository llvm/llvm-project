! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s -fopenmp-version=50 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s -fopenmp-version=50 2>&1 | FileCheck %s

! CHECK: not yet implemented: Taskloop construct
subroutine omp_taskloop
integer :: i
!$omp parallel
    !$omp taskloop
      do i = 1, 10
      !$omp cancel taskgroup
      end do
    !$omp end taskloop
!$omp end parallel
end subroutine omp_taskloop
