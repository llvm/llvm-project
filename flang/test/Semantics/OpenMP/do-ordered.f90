!RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.7.1 Ordered Clause

program omp_doOrdered
  integer:: i,j
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 2
  !BECAUSE: ORDERED clause was specified with argument 3
  !$omp do ordered(3)
  do i = 1,10
    do j = 1, 10
      print *, "hello"
    end do
  end do
  !$omp end do

  do i = 1,10
    do j = 1, 10
      !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
      !BECAUSE: ORDERED clause was specified with argument 2
      !$omp do ordered(2)
      do k = 1, 10
        print *, "hello"
      end do
      !$omp end do
    end do
  end do

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: ORDERED clause was specified with argument 2
  !$omp do ordered(2)
  do i = 1,10
    !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a worksharing-loop (or worksharing-loop SIMD) region with ORDERED clause without the parameter
    !$omp ordered
    do j = 1, 10
       print *, "hello"
    end do
    !$omp end ordered
  end do
  !$omp end do

  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 2
  !BECAUSE: ORDERED clause was specified with argument 3
  !$omp do collapse(1) ordered(3)
  do i = 1,10
    do j = 1, 10
       print *, "hello"
    end do
  end do
  !$omp end do

  !$omp parallel num_threads(4)
  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: ORDERED clause was specified with argument 2
  !$omp do ordered(2) collapse(1)
  do i = 1,10
    !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a worksharing-loop (or worksharing-loop SIMD) region with ORDERED clause without the parameter
    !$omp ordered
    do j = 1, 10
       print *, "hello"
    end do
    !$omp end ordered
  end do
  !$omp end parallel
end program omp_doOrdered
