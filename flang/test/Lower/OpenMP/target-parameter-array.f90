! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Test that parameter (constant) arrays can be mapped to OpenMP target regions
! and are mapped as read-only when accessed with dynamic indices.

module param_array_module
  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  ! Parameter arrays that should be mapped to device
  real(dp), parameter :: const_array(3) = [1.0_dp, 2.0_dp, 3.0_dp]
  integer, parameter :: int_array(4) = [10, 20, 30, 40]

contains

! Parameter array with dynamic index in target region with teams distribute
! CHECK-LABEL: func.func @_QMparam_array_modulePtest_param_array_target
subroutine test_param_array_target(idx)
  integer, intent(in) :: idx
  integer :: i
  real(dp) :: result

  ! CHECK: omp.map.info{{.*}}map_clauses(implicit, to){{.*}}{name = "const_array"}
  !$omp target teams distribute parallel do
  do i = 1, 3
    ! Access parameter array with dynamic index
    result = const_array(idx)
  end do
  !$omp end target teams distribute parallel do

end subroutine test_param_array_target

! Integer parameter array in simple target region
! CHECK-LABEL: func.func @_QMparam_array_modulePtest_int_param_array
subroutine test_int_param_array(idx)
  integer, intent(in) :: idx
  integer :: result

  ! CHECK: omp.map.info{{.*}}map_clauses(implicit, to){{.*}}{name = "int_array"}
  !$omp target
    ! Access parameter array with dynamic index
    result = int_array(idx)
  !$omp end target

end subroutine test_int_param_array

! Verify scalar parameters are NOT mapped (can be inlined)
! CHECK-LABEL: func.func @_QMparam_array_modulePtest_scalar_param
subroutine test_scalar_param()
  integer, parameter :: scalar_const = 42
  integer :: result

  ! CHECK-NOT: omp.map.info{{.*}}{name = "scalar_const"}
  !$omp target
    result = scalar_const
  !$omp end target

end subroutine test_scalar_param

! Test character scalar parameter with dynamic substring access
! CHECK-LABEL: func.func @_QMparam_array_modulePtest_char_substring
subroutine test_char_substring(start_idx, end_idx)
  integer, intent(in) :: start_idx, end_idx
  character(len=20), parameter :: char_scalar = "constant_string_data"
  character(len=10) :: result

  ! CHECK: omp.map.info{{.*}}map_clauses(implicit, to){{.*}}{name = "char_scalar"}
  !$omp target
    ! Dynamic substring access - character scalar must be mapped
    result = char_scalar(start_idx:end_idx)
  !$omp end target

end subroutine test_char_substring

! Verify character scalar with constant substring is NOT mapped
! CHECK-LABEL: func.func @_QMparam_array_modulePtest_char_const_substring
subroutine test_char_const_substring()
  character(len=20), parameter :: char_const = "constant_string_data"
  character(len=5) :: result

  ! CHECK-NOT: omp.map.info{{.*}}{name = "char_const"}
  !$omp target
    ! Constant substring access - can be inlined, no mapping needed
    result = char_const(1:5)
  !$omp end target

end subroutine test_char_const_substring

end module param_array_module
