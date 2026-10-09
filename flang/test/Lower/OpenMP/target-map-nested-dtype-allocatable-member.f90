! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Check that explicit maps of derived-type components can still get an
! implicit default mapper when the component type requires deep-copy mapping.
! The synthetic parent maps for var1/var2 are only structural containers for
! the explicit var1%b/var2%b list items and must not suppress mapper emission
! for those list items.

subroutine target_map_nested_dtype_allocatable_member
  type :: inner_type
    integer, allocatable :: a
  end type inner_type

  type :: outer_type
    type(inner_type) :: b
  end type outer_type

  type(outer_type) :: var1, var2

  !$omp target map(tofrom: var1%b, var2%b)
    var1%b%a = var2%b%a
  !$omp end target
end subroutine

! CHECK: omp.declare_mapper @{{.*}}inner_type_omp_default_mapper

! CHECK-LABEL: func.func @_QPtarget_map_nested_dtype_allocatable_member
! CHECK: %[[VAR1_B:.*]] = omp.map.info {{.*}}map_clauses(tofrom){{.*}}mapper(@{{.*}}inner_type_omp_default_mapper){{.*}}name("var1%b")
! CHECK: %[[VAR2_B:.*]] = omp.map.info {{.*}}map_clauses(tofrom){{.*}}mapper(@{{.*}}inner_type_omp_default_mapper){{.*}}name("var2%b")
! CHECK: omp.map.info {{.*}}map_clauses(storage){{.*}}members(%[[VAR1_B]] : [0] :{{.*}}){{.*}}name("var1"){{.*}}partial_map(true)
! CHECK: omp.map.info {{.*}}map_clauses(storage){{.*}}members(%[[VAR2_B]] : [0] :{{.*}}){{.*}}name("var2"){{.*}}partial_map(true)
