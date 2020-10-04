! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.9.2 taskloop Construct
! All loops associated with the taskloop construct must be perfectly nested,
! there must be no intervening code or any OpenMP directive between
! any two loops

program omp_taskloop
  integer i, j

  !$omp taskloop private(j) grainsize(500) nogroup
  do i=1, 10000
    do j=1, i
      call loop_body(i, j)
    end do
    !ERROR: Loops associated with !$omp taskloop is not perfectly nested
    !$omp single
    print *, "omp single"
    !$omp end single
  end do
  !$omp end taskloop

end program omp_taskloop
