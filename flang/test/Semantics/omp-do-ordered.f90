!RUN: %python %S/test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.7.1 Ordered Clause

program omp_doOrdered
  integer:: i,j
  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do ordered(3)
  do i = 1,10
    do j = 1, 10
      print *, "hello"
    end do
  end do
  !$omp end do

  do i = 1,10
    do j = 1, 10
      !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
      !$omp do ordered(2)
      do k = 1, 10
        print *, "hello"
      end do
      !$omp end do
    end do
  end do

  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do ordered(2)
  do i = 1,10
    !$omp ordered
    do j = 1, 10
       print *, "hello"
    end do
    !$omp end ordered
  end do
  !$omp end do

  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do collapse(1) ordered(3)
  do i = 1,10
    do j = 1, 10
       print *, "hello"
    end do
  end do
  !$omp end do

  !$omp parallel num_threads(4)
  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do ordered(2) collapse(1)
  do i = 1,10
    !$omp ordered
    do j = 1, 10
       print *, "hello"
    end do
    !$omp end ordered
  end do
  !$omp end parallel
end program omp_doOrdered
