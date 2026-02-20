! RUN: %flang -fopenmp -c %s
! Regression test for nested PARALLEL SECTIONS
! This test verifies that combined parallel-worksharing constructs
! can be properly nested as they create their own implicit parallel regions.

program test_parallel_sections_nesting
  implicit none
  integer :: i, j, n
  real :: a(10), b(10), c(10)
  
  n = 10
  
  ! Test 1: PARALLEL SECTIONS nesting PARALLEL SECTIONS
  !$OMP PARALLEL SECTIONS
    !$OMP SECTION
      !$OMP PARALLEL SECTIONS
        !$OMP SECTION
          do i = 1, n
            a(i) = real(i)
          end do
        !$OMP SECTION
          do j = 1, n
            b(j) = real(j * 2)
          end do
      !$OMP END PARALLEL SECTIONS
    !$OMP SECTION
      !$OMP PARALLEL DO
        do i = 1, n
          c(i) = a(i) + b(i)
        end do
      !$OMP END PARALLEL DO
  !$OMP END PARALLEL SECTIONS
  
  ! Test 2: PARALLEL SECTIONS inside PARALLEL
  !$OMP PARALLEL
    !$OMP PARALLEL SECTIONS
      !$OMP SECTION
        do i = 1, n
          a(i) = a(i) * 2
        end do
      !$OMP SECTION
        do j = 1, n
          b(j) = b(j) * 2
        end do
    !$OMP END PARALLEL SECTIONS
  !$OMP END PARALLEL
  
  ! Test 3: PARALLEL DO inside PARALLEL SECTIONS
  !$OMP PARALLEL SECTIONS
    !$OMP SECTION
      !$OMP PARALLEL DO
        do i = 1, n
          c(i) = c(i) + a(i)
        end do
      !$OMP END PARALLEL DO
    !$OMP SECTION
      c = c + 1.0
  !$OMP END PARALLEL SECTIONS
  
end program test_parallel_sections_nesting