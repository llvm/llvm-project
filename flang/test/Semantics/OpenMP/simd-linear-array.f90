! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
! OpenMP Version 5.2
! Test that arrays in LINEAR clause are rejected on SIMD directive
! This test addresses issue #171007 - crash with array in LINEAR clause

subroutine test_1d_array_in_linear()
  implicit none
  integer :: j, arr(2)
  
  !ERROR: List item 'arr' in LINEAR clause must be a scalar variable
  !$omp simd linear(arr)
  do j=1,10
  end do
end subroutine

subroutine test_multidim_array()
  implicit none
  integer :: j, matrix(3,3)
  
  !ERROR: List item 'matrix' in LINEAR clause must be a scalar variable
  !$omp simd linear(matrix)
  do j=1,10
  end do
end subroutine

subroutine test_assumed_shape_array(arr)
  implicit none
  integer :: j
  integer, intent(in) :: arr(:)
  
  !ERROR: List item 'arr' in LINEAR clause must be a scalar variable
  !$omp simd linear(arr)
  do j=1,10
  end do
end subroutine

subroutine test_multiple_vars_with_array()
  implicit none
  integer :: j, scalar1, arr(5), scalar2
  
  !ERROR: List item 'arr' in LINEAR clause must be a scalar variable
  !$omp simd linear(scalar1, arr, scalar2)
  do j=1,10
    scalar1 = j
    scalar2 = j
  end do
end subroutine

! Valid case - scalar should work fine
subroutine test_scalar_valid()
  implicit none
  integer :: j, scalar
  
  !$omp simd linear(scalar)
  do j=1,10
    scalar = j
  end do
end subroutine

! Valid case - multiple scalars should work
subroutine test_multiple_scalars_valid()
  implicit none
  integer :: j, scalar1, scalar2, scalar3
  
  !$omp simd linear(scalar1, scalar2, scalar3)
  do j=1,10
    scalar1 = j
    scalar2 = j
    scalar3 = j
  end do
end subroutine

! Valid case - declare simd with REF modifier allows arrays
subroutine test_declare_simd_ref_array_valid(arr)
  implicit none
  integer, intent(in) :: arr(:)
  
  !$omp declare simd linear(ref(arr))
  ! No error expected - REF modifier allows assumed-shape arrays
end subroutine
