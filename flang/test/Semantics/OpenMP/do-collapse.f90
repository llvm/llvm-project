!RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.7.1 Collapse Clause
program omp_doCollapse
  integer:: i,j
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 2
  !BECAUSE: COLLAPSE clause was specified with argument 3
  !$omp do collapse(3)
  do i = 1,10
    do j = 1, 10
      print *, "hello"
    end do
  end do
  !$omp end do

  do i = 1,10
    do j = 1, 10
      !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
      !BECAUSE: COLLAPSE clause was specified with argument 2
      !$omp do collapse(2)
      do k = 1, 10
        print *, "hello"
      end do
      !$omp end do
    end do
  end do

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp parallel do collapse(2)
    do i = 1, 3
      !BECAUSE: DO loop without loop control is not a valid affected loop
      !ERROR: Loop control is not present in the DO LOOP
      do
      end do
    end do

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !ERROR: At most one COLLAPSE clause can appear on the SIMD directive
  !$omp simd collapse(2) collapse(1)
  do i = 1, 4
    j = j + i + 1
  end do
  !$omp end simd
end program omp_doCollapse
