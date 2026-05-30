! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
! Verify that array sections and array elements in LINEAR clause are rejected.
! The LINEAR clause requires a scalar variable name (OpenMP 4.5 §2.15.3.7).

subroutine test_array_section_in_linear()
  implicit none
  integer :: i
  integer, dimension(0:99, -99:10, 200) :: a, b, c

  !$omp parallel
    !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear in a LINEAR clause
    !$omp do linear(a(:,1,1))
    do i = 0, 99
      c(i, 1, 1) = a(i, 1, 1) + b(i, 1, 1)
    end do
    !$omp end do
  !$omp end parallel
end subroutine

subroutine test_array_element_in_linear()
  implicit none
  integer :: i
  integer :: arr(10)

  !$omp parallel
    !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear in a LINEAR clause
    !$omp do linear(arr(1))
    do i = 1, 10
      arr(i) = i
    end do
    !$omp end do
  !$omp end parallel
end subroutine

subroutine test_structure_component_in_linear()
  implicit none
  integer :: i
  type :: my_type
    integer :: field
  end type
  type(my_type) :: obj

  !$omp parallel
    !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear in a LINEAR clause
    !$omp do linear(obj%field)
    do i = 1, 10
    end do
    !$omp end do
  !$omp end parallel
end subroutine

subroutine test_valid_scalar_in_linear()
  implicit none
  integer :: i, j

  !$omp parallel
    !$omp do linear(j)
    do i = 1, 10
      j = i
    end do
    !$omp end do
  !$omp end parallel
end subroutine

