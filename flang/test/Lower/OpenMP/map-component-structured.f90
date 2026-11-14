! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

module component_structured_types
  type :: item
    integer :: before
    real, allocatable :: a(:)
    integer :: after
  end type
end module

! The parent descriptor is needed by the kernel, but its pointee storage is
! an implicit container for the explicitly mapped component. A smaller map
! of that component may already exist from target enter data.
! CHECK-LABEL: func.func @_QParray_target(
! CHECK: %[[W:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFarray_targetEw")
! CHECK: omp.map.info {{.*}}map_clauses(present, tofrom) {{.*}}var_ptr_ptr(
! CHECK: omp.map.info var_ptr(%[[W]]#1 : {{.*}}) map_clauses(implicit, present, storage) {{.*}}var_ptr_ptr(
! CHECK: %[[DESC:.*]] = omp.map.info var_ptr(%[[W]]#1 : {{.*}}) map_clauses(always, to)
! CHECK: omp.target {{.*}}map_entries(%[[DESC]] ->
subroutine array_target(w, i)
  use component_structured_types
  type(item), allocatable :: w(:)
  integer :: i
  !$omp target map(present, tofrom: w(i)%a)
    w(i)%a = 7.0
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QParray_target_data(
! CHECK: %[[W:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFarray_target_dataEw")
! CHECK: omp.map.info {{.*}}map_clauses(present, tofrom) {{.*}}var_ptr_ptr(
! CHECK: omp.map.info var_ptr(%[[W]]#1 : {{.*}}) map_clauses(implicit, present, storage) {{.*}}var_ptr_ptr(
! CHECK: %[[DESC:.*]] = omp.map.info var_ptr(%[[W]]#1 : {{.*}}) map_clauses(always, to)
! CHECK: omp.target_data map_entries(%[[DESC]]
subroutine array_target_data(w, i)
  use component_structured_types
  type(item), allocatable :: w(:)
  integer :: i
  !$omp target data map(present, tofrom: w(i)%a)
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPscalar_target(
! CHECK: %[[W:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFscalar_targetEw")
! CHECK: omp.map.info {{.*}}map_clauses(present, tofrom) {{.*}}var_ptr_ptr(
! CHECK: omp.map.info var_ptr(%[[W]]#1 : {{.*}}) map_clauses(implicit, present, storage) {{.*}}var_ptr_ptr(
! CHECK: %[[DESC:.*]] = omp.map.info var_ptr(%[[W]]#1 : {{.*}}) map_clauses(always, to)
! CHECK: omp.target {{.*}}map_entries(%[[DESC]] ->
subroutine scalar_target(w)
  use component_structured_types
  type(item), allocatable :: w
  !$omp target map(present, tofrom: w%a)
    w%a = 7.0
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPscalar_target_data(
! CHECK: %[[W:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFscalar_target_dataEw")
! CHECK: omp.map.info {{.*}}map_clauses(present, tofrom) {{.*}}var_ptr_ptr(
! CHECK: omp.map.info var_ptr(%[[W]]#1 : {{.*}}) map_clauses(implicit, present, storage) {{.*}}var_ptr_ptr(
! CHECK: %[[DESC:.*]] = omp.map.info var_ptr(%[[W]]#1 : {{.*}}) map_clauses(always, to)
! CHECK: omp.target_data map_entries(%[[DESC]]
subroutine scalar_target_data(w)
  use component_structured_types
  type(item), allocatable :: w
  !$omp target data map(present, tofrom: w%a)
  !$omp end target data
end subroutine

! Assumed-shape parents need normal descriptor mapping and attachment with
! the element bounds. Privatization would look up the unmapped array base.
! CHECK-LABEL: func.func @_QPassumed_shape_target(
! CHECK: %[[BOX:.*]] = fir.alloca !fir.box<!fir.array<
! CHECK: omp.map.info {{.*}}map_clauses(present, tofrom) {{.*}}var_ptr_ptr(
! CHECK: omp.map.info var_ptr(%[[BOX]] : {{.*}}) map_clauses(implicit, present, storage) {{.*}}bounds(%[[BOUNDS:.*]]) name("w(i)")
! CHECK: %[[DESC:.*]] = omp.map.info var_ptr(%[[BOX]] : {{.*}}) map_clauses(always, to) capture(ByRef) members(
! CHECK: omp.map.info var_ptr(%[[BOX]] : {{.*}}) map_clauses(attach, ref_ptr, ref_ptee) {{.*}}bounds(%[[BOUNDS]])
! CHECK: omp.target {{.*}}map_entries(%[[DESC]] ->
subroutine assumed_shape_target(w, i)
  use component_structured_types
  type(item) :: w(:)
  integer :: i
  !$omp target map(present, tofrom: w(i)%a)
    w(i)%a = 7.0
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPassumed_shape_target_data(
! CHECK: %[[BOX:.*]] = fir.alloca !fir.box<!fir.array<
! CHECK: omp.map.info {{.*}}map_clauses(present, tofrom) {{.*}}var_ptr_ptr(
! CHECK: omp.map.info var_ptr(%[[BOX]] : {{.*}}) map_clauses(implicit, present, storage) {{.*}}bounds(%[[BOUNDS:.*]]) name("w(i)")
! CHECK: %[[DESC:.*]] = omp.map.info var_ptr(%[[BOX]] : {{.*}}) map_clauses(always, to) capture(ByRef) members(
! CHECK: omp.map.info var_ptr(%[[BOX]] : {{.*}}) map_clauses(attach, ref_ptr, ref_ptee) {{.*}}bounds(%[[BOUNDS]])
! CHECK: omp.target_data map_entries(%[[DESC]]
subroutine assumed_shape_target_data(w, i)
  use component_structured_types
  type(item) :: w(:)
  integer :: i
  !$omp target data map(present, tofrom: w(i)%a)
  !$omp end target data
end subroutine

! An explicit parent storage map must not acquire implicit-map semantics.
! CHECK-LABEL: func.func @_QPexplicit_parent_target(
! CHECK: %[[W:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFexplicit_parent_targetEw")
! CHECK: omp.map.info var_ptr(%[[W]]#1 : {{.*}}) map_clauses(storage) {{.*}}var_ptr_ptr(
! CHECK: %[[DESC:.*]] = omp.map.info var_ptr(%[[W]]#1 : {{.*}}) map_clauses(always, to)
! CHECK: omp.target {{.*}}map_entries(%[[DESC]] ->
subroutine explicit_parent_target(w)
  use component_structured_types
  type(item), allocatable :: w
  !$omp target map(alloc: w) map(tofrom: w%a)
    w%a = 7.0
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPexplicit_parent_target_data(
! CHECK: %[[W:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFexplicit_parent_target_dataEw")
! CHECK: omp.map.info var_ptr(%[[W]]#1 : {{.*}}) map_clauses(storage) {{.*}}var_ptr_ptr(
! CHECK: %[[DESC:.*]] = omp.map.info var_ptr(%[[W]]#1 : {{.*}}) map_clauses(always, to)
! CHECK: omp.target_data map_entries(%[[DESC]]
subroutine explicit_parent_target_data(w)
  use component_structured_types
  type(item), allocatable :: w
  !$omp target data map(alloc: w) map(tofrom: w%a)
  !$omp end target data
end subroutine
