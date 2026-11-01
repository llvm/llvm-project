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

  ! THREADS+SIMD inside a plain SIMD (not DO SIMD) region must be rejected
  !$omp simd
  do i = 1, N
    !ERROR: An ORDERED directive with SIMD and THREADS clauses must be closely nested in a worksharing-loop SIMD region
    !$omp ordered threads simd
    arrayA(i) = foo(i)
    !$omp end ordered
  end do
  !$omp end simd

  ! THREADS+SIMD inside a DO SIMD region is valid
  !$omp do simd ordered
  do i = 1, N
    !$omp ordered threads simd
    arrayA(i) = foo(i)
    !$omp end ordered
  end do
  !$omp end do simd
end

! No diagnostic when ORDERED THREADS SIMD is in a called routine (enclosing
! DO SIMD region is not visible to the static checker).
subroutine contains_ordered_threads_simd()
  !$omp ordered threads simd
  !$omp end ordered
end subroutine

subroutine sub2()
  integer :: i, N = 10
  external :: contains_ordered_threads_simd
  !$omp do simd ordered
  do i = 1, N
    call contains_ordered_threads_simd()
  end do
  !$omp end do simd
end subroutine
