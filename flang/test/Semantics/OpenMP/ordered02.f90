! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.19.9 Ordered Construct

subroutine sub1()
  integer :: i, j, N = 10
  real :: arrayA(10), arrayB(10)
  real, external :: foo, bar

  !$omp ordered
  arrayA(i) = foo(i)
  !$omp end ordered

  !$omp ordered threads
  arrayA(i) = foo(i)
  !$omp end ordered

  !$omp ordered simd
  arrayA(i) = foo(i)
  !$omp end ordered

  !$omp sections
  do i = 1, N
    !$omp ordered
    arrayA(i) = foo(i)
    !$omp end ordered
  end do
  !$omp end sections

  !$omp do ordered
  do i = 1, N
    arrayB(i) = bar(i)
    !$omp ordered
    arrayA(i) = foo(i)
    !$omp end ordered
  end do
  !$omp end do

  !$omp sections
  do i = 1, N
    !ERROR: An ORDERED directive with SIMD clause must be closely nested in a SIMD or worksharing-loop SIMD region
    !$omp ordered simd
    arrayA(i) = foo(i)
    !$omp end ordered
  end do
  !$omp end sections

  !$omp do ordered
  do i = 1, N
    !$omp parallel
    do j = 1, N
      !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a SIMD, worksharing-loop, or worksharing-loop SIMD region
      !$omp ordered
      arrayA(i) = foo(i)
      !$omp end ordered
    end do
    !$omp end parallel
  end do
  !$omp end do

  !$omp do ordered
  do i = 1, N
    !$omp target parallel
    do j = 1, N
      !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a SIMD, worksharing-loop, or worksharing-loop SIMD region
      !$omp ordered
      arrayA(i) = foo(i)
      !$omp end ordered
    end do
    !$omp end target parallel
  end do
  !$omp end do

  !$omp do
  do i = 1, N
    !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a worksharing-loop (or worksharing-loop SIMD) region with ORDERED clause without the parameter
    !$omp ordered
    arrayA(i) = foo(i)
    !$omp end ordered
  end do
  !$omp end do

  !$omp do
  do i = 1, N
    !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a worksharing-loop (or worksharing-loop SIMD) region with ORDERED clause without the parameter
    !$omp ordered threads
    arrayA(i) = foo(i)
    !$omp end ordered
  end do
  !$omp end do

  !$omp do ordered(1)
  do i = 1, N
    !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a worksharing-loop (or worksharing-loop SIMD) region with ORDERED clause without the parameter
    !$omp ordered
    arrayA(i) = foo(i)
    !$omp end ordered
  end do
  !$omp end do

  !$omp do ordered(1)
  do i = 1, N
    !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a worksharing-loop (or worksharing-loop SIMD) region with ORDERED clause without the parameter
    !$omp ordered threads
    arrayA(i) = foo(i)
    !$omp end ordered
  end do
  !$omp end do

  !$omp parallel do ordered(1)
  do i = 1, N
    !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a worksharing-loop (or worksharing-loop SIMD) region with ORDERED clause without the parameter
    !$omp ordered
    arrayA(i) = foo(i)
    !$omp end ordered
  end do
  !$omp end parallel do

  !$omp parallel do ordered(1)
  do i = 1, N
    !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a worksharing-loop (or worksharing-loop SIMD) region with ORDERED clause without the parameter
    !$omp ordered threads
    arrayA(i) = foo(i)
    !$omp end ordered
  end do
  !$omp end parallel do

  !$omp target parallel do ordered(1)
  do i = 1, N
    !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a worksharing-loop (or worksharing-loop SIMD) region with ORDERED clause without the parameter
    !$omp ordered
    arrayA(i) = foo(i)
    !$omp end ordered
  end do
  !$omp end target parallel do

  !$omp target parallel do ordered(1)
  do i = 1, N
    !ERROR: An ORDERED directive without the DEPEND clause must be closely nested in a worksharing-loop (or worksharing-loop SIMD) region with ORDERED clause without the parameter
    !$omp ordered threads
    arrayA(i) = foo(i)
    !$omp end ordered
  end do
  !$omp end target parallel do
end
