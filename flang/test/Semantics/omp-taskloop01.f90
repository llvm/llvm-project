! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! OpenMP Version 4.5
! 2.9.2 taskloop Construct

subroutine parallel_work
  integer i
  integer j

  !$omp taskgroup
    !$omp task
    call long_running_task()
    !$omp end task

    !$omp taskloop private(j) grainsize(500) nogroup
    do i=1,10000
      do j=1,i
        call loop_body(i, j)
      end do
    end do
    !$omp end taskloop
  !$omp end taskgroup

end subroutine parallel_work
