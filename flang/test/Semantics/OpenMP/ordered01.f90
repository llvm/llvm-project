! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.19.9 Ordered Construct

program main
  integer :: i, N = 10
  real :: a, arrayA(10), arrayB(10), arrayC(10)
  real, external :: foo, bar, baz

  !$omp do ordered
  do i = 1, N
    !ERROR: At most one THREADS clause can appear on the ORDERED directive
    !$omp ordered threads threads
    arrayA(i) = i
    !$omp end ordered
  end do
  !$omp end do

  !$omp simd
  do i = 1, N
    !ERROR: At most one SIMD clause can appear on the ORDERED directive
    !$omp ordered simd simd
    arrayA(i) = i
    !$omp end ordered
  end do
  !$omp end simd

  !$omp do simd ordered
  do i = 1, N
    !ERROR: At most one SIMD clause can appear on the ORDERED directive
    !$omp ordered simd simd
    arrayA(i) = i
    !$omp end ordered
  end do
  !$omp end do simd

  !$omp do ordered(1)
  do i = 2, N
    !ERROR: Only SINK or SOURCE dependence types are allowed when ORDERED construct is a standalone construct with no ORDERED region
    !ERROR: At most one SOURCE dependence type can appear on the ORDERED directive
    !$omp ordered depend(source) depend(inout: arrayA) depend(source)
    arrayA(i) = foo(i)
    !ERROR: The SINK and SOURCE dependence types are mutually exclusive
    !ERROR: At most one SOURCE dependence type can appear on the ORDERED directive
    !$omp ordered depend(sink: i - 1) depend(source) depend(source)
    arrayB(i) = bar(arrayA(i), arrayB(i-1))
    !ERROR: Only SINK or SOURCE dependence types are allowed when ORDERED construct is a standalone construct with no ORDERED region
    !ERROR: Only SINK or SOURCE dependence types are allowed when ORDERED construct is a standalone construct with no ORDERED region
    !$omp ordered depend(out: arrayC) depend(in: arrayB)
    arrayC(i) = baz(arrayB(i-1))
  end do
  !$omp end do

  !$omp do ordered(1)
  do i = 2, N
    !ERROR: DEPEND clauses are not allowed when ORDERED construct is a block construct with an ORDERED region
    !$omp ordered depend(source)
    arrayA(i) = foo(i)
    !$omp end ordered
    !ERROR: DEPEND clauses are not allowed when ORDERED construct is a block construct with an ORDERED region
    !$omp ordered depend(sink: i - 1)
    arrayB(i) = bar(arrayA(i), arrayB(i-1))
    !$omp end ordered
  end do
  !$omp end do

contains
  subroutine work1()
    !ERROR: THREADS and SIMD clauses are not allowed when ORDERED construct is a standalone construct with no ORDERED region
    !$omp ordered simd
  end subroutine work1

  subroutine work2()
    !ERROR: THREADS and SIMD clauses are not allowed when ORDERED construct is a standalone construct with no ORDERED region
    !$omp ordered threads
  end subroutine work2

end program main
