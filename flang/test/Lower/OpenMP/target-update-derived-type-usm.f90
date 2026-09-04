! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa %s -o - | FileCheck %s

! Verify that unified shared memory keeps the regular target update because a
! packed transfer and target region would add overhead to directly accessible
! storage.

module target_update_derived_type_usm
  !$omp requires unified_shared_memory
  type :: aggregate
    real(8) :: first
    real(8) :: gap
    integer :: last
  end type
contains

! CHECK-LABEL: func.func @_QMtarget_update_derived_type_usmPupdate(
subroutine update(value)
  type(aggregate) :: value

  ! CHECK: %[[FIRST_MAP:.*]] = omp.map.info {{.*}} map_clauses(to)
  ! CHECK: %[[LAST_MAP:.*]] = omp.map.info {{.*}} map_clauses(to)
  ! CHECK-NOT: omp.target kernel_type(generic)
  ! CHECK: omp.target_update map_entries(%[[FIRST_MAP]], %[[LAST_MAP]]
  !$omp target update to(value%first, value%last)
end subroutine

end module
