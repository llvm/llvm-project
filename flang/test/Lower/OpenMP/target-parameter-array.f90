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

! Test 1: Parameter array with dynamic index in target region with teams distribute
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

end module param_array_module
