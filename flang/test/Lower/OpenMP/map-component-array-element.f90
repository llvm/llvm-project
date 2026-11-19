! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

module component_types
  type :: item
    real, allocatable :: a(:)
    integer :: k
  end type
end module

! The synthesized parent on mapping-only constructs must be the scalar
! record, so only the component's storage is mapped. Keeping the containing
! array descriptor here would also map the whole record and the array box.

! CHECK-LABEL: func.func @_QPenter_component(
! CHECK-NOT: omp.map.info{{.*}}!fir.box<!fir.heap<!fir.array<?x!fir.type
! CHECK: %[[ENTER_PARENT:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.type<_QMcomponent_typesTitem
! CHECK-SAME: map_clauses(storage)
! CHECK-SAME: partial_map(true)
! CHECK-NOT: omp.map.info{{.*}}!fir.box<!fir.heap<!fir.array<?x!fir.type
! CHECK: omp.target_enter_data map_entries(%[[ENTER_PARENT]]
subroutine enter_component(w, i)
  use component_types
  type(item), allocatable :: w(:)
  integer :: i
  !$omp target enter data map(to: w(i)%a)
end subroutine

! The same rule applies to an allocatable scalar record. Its address comes
! from fir.box_addr, which must not cause descriptor finalization to restore
! a map of the parent descriptor.
! CHECK-LABEL: func.func @_QPmap_scalar_component(
! CHECK: %[[SCALAR_ENTER:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.heap<!fir.type<_QMcomponent_typesTitem
! CHECK-SAME: map_clauses(storage)
! CHECK-SAME: partial_map(true)
! CHECK: omp.target_enter_data map_entries(%[[SCALAR_ENTER]]
! CHECK: %[[SCALAR_EXIT:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.heap<!fir.type<_QMcomponent_typesTitem
! CHECK-SAME: map_clauses(storage)
! CHECK-SAME: partial_map(true)
! CHECK: omp.target_exit_data map_entries(%[[SCALAR_EXIT]]
subroutine map_scalar_component(w)
  use component_types
  type(item), allocatable :: w
  !$omp target enter data map(to: w%a)
  !$omp target exit data map(delete: w%a)
end subroutine

! CHECK-LABEL: func.func @_QPexit_component(
! CHECK-NOT: omp.map.info{{.*}}!fir.box<!fir.heap<!fir.array<?x!fir.type
! CHECK: %[[EXIT_PARENT:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.type<_QMcomponent_typesTitem
! CHECK-SAME: map_clauses(storage)
! CHECK-SAME: partial_map(true)
! CHECK-NOT: omp.map.info{{.*}}!fir.box<!fir.heap<!fir.array<?x!fir.type
! CHECK: omp.target_exit_data map_entries(%[[EXIT_PARENT]]
subroutine exit_component(w, i)
  use component_types
  type(item), allocatable :: w(:)
  integer :: i
  !$omp target exit data map(delete: w(i)%a)
end subroutine
