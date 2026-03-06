!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! This should compile without errors. Check for a symptom of a reasonable
! output.

!CHECK: omp.task depend

subroutine omp_task_depend_reproducer(work, myid, shift)
  implicit none
  integer, intent(in) :: myid, shift
  real, intent(inout) :: work(*)

!$omp parallel shared(work, myid, shift)
  !$omp single
    !$omp task depend(in:work(myid+shift-1)) depend(in:work(myid-1)) depend(out:work(myid))
      call dummy_kernel(work(myid))
    !$omp end task
  !$omp end single
!$omp end parallel
contains
  subroutine dummy_kernel(x)
    real :: x
    x = x + 1.0
  end subroutine dummy_kernel
end subroutine omp_task_depend_reproducer
