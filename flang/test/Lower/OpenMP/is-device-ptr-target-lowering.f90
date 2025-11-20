! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Verify that lowering a TARGET with is_device_ptr attaches
! the clause to the resulting omp.target op.

program test_is_device_ptr_lowering
  use iso_c_binding, only: c_associated, c_ptr
  implicit none
  integer :: i
  integer :: arr(4)
  type(c_ptr) :: p

  i = 0
  arr = 0

  !$omp target is_device_ptr(p)
    if (c_associated(p)) i = i + 1
    arr(1) = i
  !$omp end target
end program test_is_device_ptr_lowering

! CHECK: %[[P_STORAGE:.*]] = omp.map.info {{.*}}{name = "p"}
! CHECK: %[[P_IS:.*]] = omp.map.info {{.*}}{name = "p"}
! CHECK: %[[I_MAP:.*]] = omp.map.info {{.*}}{name = "i"}
! CHECK: %[[ARR_MAP:.*]] = omp.map.info {{.*}}{name = "arr"}
! CHECK: omp.target is_device_ptr(%[[P_IS]] :
! CHECK-SAME: has_device_addr(%[[P_STORAGE]] ->
! CHECK-SAME: map_entries(%[[I_MAP]] ->
! CHECK-SAME: %[[ARR_MAP]] ->
! CHECK: omp.terminator
