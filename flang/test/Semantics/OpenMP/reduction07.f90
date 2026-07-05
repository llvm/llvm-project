! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause
program omp_reduction

  integer :: a,i,j,l
  integer :: k = 10
  !$omp parallel private(k)
  !ERROR: REDUCTION variable 'k' is PRIVATE in outer context must be shared in the parallel regions to which any of the worksharing regions arising from the worksharing construct bind.
  !$omp do reduction(+:k)
  do i = 1, 10
    k = k + 1
  end do
  !$omp end do
  !$omp end parallel


  !$omp parallel private(j),reduction(+:k)
  !ERROR: REDUCTION variable 'k' is REDUCTION in outer context must be shared in the parallel regions to which any of the worksharing regions arising from the worksharing construct bind.
  !$omp do reduction(+:k)
  do i = 1, 10
    k = k + 1
  end do
  !$omp end do
  !$omp end parallel

  !$omp parallel private(j),firstprivate(k)
  !ERROR: REDUCTION variable 'k' is FIRSTPRIVATE in outer context must be shared in the parallel regions to which any of the worksharing regions arising from the worksharing construct bind.
  !$omp do reduction(min:k)
  do i = 1, 10
    k = k + 1
  end do
  !$omp end do
  !$omp end parallel


  !$omp parallel private(l,j),firstprivate(k)
  !ERROR: REDUCTION variable 'k' is FIRSTPRIVATE in outer context must be shared in the parallel regions to which any of the worksharing regions arising from the worksharing construct bind.
  !ERROR: REDUCTION variable 'j' is PRIVATE in outer context must be shared in the parallel regions to which any of the worksharing regions arising from the worksharing construct bind.
  !$omp sections reduction(ior:k) reduction(*:j)
  do i = 1, 10
    k = ior(k, 1)
    j = j * 3
  end do
  !$omp end sections
  !$omp end parallel

!$omp sections private(k)
  !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
  !ERROR: REDUCTION variable 'k' is PRIVATE in outer context must be shared in the parallel regions to which any of the worksharing regions arising from the worksharing construct bind.
  !$omp do reduction(+:k) reduction(max:j)
  do i = 1, 10
    k = k + 1
  end do
  !$omp end do
!$omp end sections

!$omp sections private(k)
  !$omp target
  do j = 1,10
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp do reduction(+:k) reduction(max:j)
    do i = 1, 10
      k = k + 1
    end do
    !$omp end do
  end do
  !$omp end target
!$omp end sections

!$omp parallel reduction(+:a)
!ERROR: REDUCTION variable 'a' is REDUCTION in outer context must be shared in the parallel regions to which any of the worksharing regions arising from the worksharing construct bind.
!$omp sections reduction(*:a)
a = a + 10
!$omp end sections
!$omp end parallel

!$omp parallel reduction(*:a)
!$omp end parallel

!$omp parallel reduction(ieor:a)
!ERROR: REDUCTION variable 'a' is REDUCTION in outer context must be shared in the parallel regions to which any of the worksharing regions arising from the worksharing construct bind.
!$omp sections reduction(+:a)
a = ieor(a, 10)
!$omp end sections
!$omp end parallel

!$omp parallel private(a)
!$omp parallel reduction(ieor:a)
!$omp end parallel
!$omp end parallel

!$omp task firstprivate(a)
!$omp parallel do reduction(+:a)
do i=1,10
  a=a+j
end do
!$omp end parallel do
!$omp end task

end program omp_reduction
