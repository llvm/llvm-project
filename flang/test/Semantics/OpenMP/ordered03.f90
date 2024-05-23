! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.19.9 Ordered Construct

subroutine sub1()
  integer :: i, j, N = 10
  real :: arrayA(10), arrayB(10)
  real, external :: foo, bar

  !$omp do ordered(1)
  do i = 1, N
    !$omp ordered depend(source)
    arrayA(i) = foo(i)
    !$omp ordered depend(sink: i - 1)
    arrayB(i) = bar(i - 1)
  end do
  !$omp end do

  !$omp do ordered(1)
  do i = 1, N
    !$omp target
    do j = 1, N
      !ERROR: An ORDERED construct with the DEPEND clause must be closely nested in a worksharing-loop (or parallel worksharing-loop) construct with ORDERED clause with a parameter
      !$omp ordered depend(source)
      arrayA(i) = foo(i)
      !ERROR: An ORDERED construct with the DEPEND clause must be closely nested in a worksharing-loop (or parallel worksharing-loop) construct with ORDERED clause with a parameter
      !$omp ordered depend(sink: i - 1)
      arrayB(i) = bar(i - 1)
    end do
    !$omp end target
  end do
  !$omp end do

  !$omp target
  !$omp parallel do ordered(1)
  do i = 1, N
    !$omp ordered depend(source)
    arrayA(i) = foo(i)
    !$omp ordered depend(sink: i - 1)
    arrayB(i) = bar(i - 1)
  end do
  !$omp end parallel do
  !$omp end target

  !$omp target parallel do ordered(1)
  do i = 1, N
    !$omp ordered depend(source)
    arrayA(i) = foo(i)
    !$omp ordered depend(sink: i - 1)
    arrayB(i) = bar(i - 1)
  end do
  !$omp end target parallel do

  !$omp target teams distribute parallel do ordered(1)
  do i = 1, N
    !ERROR: An ORDERED construct with the DEPEND clause must be closely nested in a worksharing-loop (or parallel worksharing-loop) construct with ORDERED clause with a parameter
    !$omp ordered depend(source)
    arrayA(i) = foo(i)
    !ERROR: An ORDERED construct with the DEPEND clause must be closely nested in a worksharing-loop (or parallel worksharing-loop) construct with ORDERED clause with a parameter
    !$omp ordered depend(sink: i - 1)
    arrayB(i) = bar(i - 1)
  end do
  !$omp end target teams distribute parallel do

  !$omp do ordered
  do i = 1, N
    !ERROR: An ORDERED construct with the DEPEND clause must be closely nested in a worksharing-loop (or parallel worksharing-loop) construct with ORDERED clause with a parameter
    !$omp ordered depend(source)
    arrayA(i) = foo(i)
    !ERROR: An ORDERED construct with the DEPEND clause must be closely nested in a worksharing-loop (or parallel worksharing-loop) construct with ORDERED clause with a parameter
    !$omp ordered depend(sink: i - 1)
    arrayB(i) = bar(i - 1)
  end do
  !$omp end do

  !$omp parallel do ordered
  do i = 1, N
    !ERROR: An ORDERED construct with the DEPEND clause must be closely nested in a worksharing-loop (or parallel worksharing-loop) construct with ORDERED clause with a parameter
    !$omp ordered depend(source)
    arrayA(i) = foo(i)
    !ERROR: An ORDERED construct with the DEPEND clause must be closely nested in a worksharing-loop (or parallel worksharing-loop) construct with ORDERED clause with a parameter
    !$omp ordered depend(sink: i - 1)
    arrayB(i) = bar(i - 1)
  end do
  !$omp end parallel do

  !$omp target parallel do ordered
  do i = 1, N
    !ERROR: An ORDERED construct with the DEPEND clause must be closely nested in a worksharing-loop (or parallel worksharing-loop) construct with ORDERED clause with a parameter
    !$omp ordered depend(source)
    arrayA(i) = foo(i)
    !ERROR: An ORDERED construct with the DEPEND clause must be closely nested in a worksharing-loop (or parallel worksharing-loop) construct with ORDERED clause with a parameter
    !$omp ordered depend(sink: i - 1)
    arrayB(i) = bar(i - 1)
  end do
  !$omp end target parallel do

  !$omp do ordered(1)
  do i = 1, N
    !ERROR: The number of variables in DEPEND(SINK: vec) clause does not match the parameter specified in ORDERED clause
    !$omp ordered depend(sink: i - 1) depend(sink: i - 1, j)
    arrayB(i) = bar(i - 1, j)
  end do
  !$omp end do

  !$omp do ordered(2)
  do i = 1, N
    do j = 1, N
      !ERROR: The number of variables in DEPEND(SINK: vec) clause does not match the parameter specified in ORDERED clause
      !$omp ordered depend(sink: i - 1) depend(sink: i - 1, j)
      arrayB(i) = foo(i - 1) + bar(i - 1, j)
    end do
  end do
  !$omp end do

  !ERROR: An ORDERED construct with the DEPEND clause must be closely nested in a worksharing-loop (or parallel worksharing-loop) construct with ORDERED clause with a parameter
  !$omp ordered depend(source)

  !ERROR: An ORDERED construct with the DEPEND clause must be closely nested in a worksharing-loop (or parallel worksharing-loop) construct with ORDERED clause with a parameter
  !$omp ordered depend(sink: i - 1)
end
