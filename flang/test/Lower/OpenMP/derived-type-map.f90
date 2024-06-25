!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s


!CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.type<_QFmaptype_derived_implicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}> {bindc_name = "scalar_arr", uniq_name = "_QFmaptype_derived_implicitEscalar_arr"}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "_QFmaptype_derived_implicitEscalar_arr"} : (!fir.ref<!fir.type<_QFmaptype_derived_implicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>) -> (!fir.ref<!fir.type<_QFmaptype_derived_implicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.ref<!fir.type<_QFmaptype_derived_implicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>)
!CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.type<_QFmaptype_derived_implicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.type<_QFmaptype_derived_implicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>) map_clauses(implicit, tofrom) capture(ByRef) -> !fir.ref<!fir.type<_QFmaptype_derived_implicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>> {name = "scalar_arr"}
!CHECK:     omp.target map_entries(%[[MAP]] -> %[[ARG0:.*]] : !fir.ref<!fir.type<_QFmaptype_derived_implicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>) {
!CHECK:         ^bb0(%[[ARG0]]: !fir.ref<!fir.type<_QFmaptype_derived_implicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>):
subroutine mapType_derived_implicit
    type :: scalar_and_array
      real(4) :: real
      integer(4) :: array(10)
      integer(4) :: int
    end type scalar_and_array
    type(scalar_and_array) :: scalar_arr 
    
    !$omp target
       scalar_arr%int = 1
    !$omp end target
end subroutine mapType_derived_implicit

!CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.type<_QFmaptype_derived_explicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}> {bindc_name = "scalar_arr", uniq_name = "_QFmaptype_derived_explicitEscalar_arr"}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "_QFmaptype_derived_explicitEscalar_arr"} : (!fir.ref<!fir.type<_QFmaptype_derived_explicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>) -> (!fir.ref<!fir.type<_QFmaptype_derived_explicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.ref<!fir.type<_QFmaptype_derived_explicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>)
!CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[DECLARE]]#0 : !fir.ref<!fir.type<_QFmaptype_derived_explicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.type<_QFmaptype_derived_explicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.type<_QFmaptype_derived_explicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>> {name = "scalar_arr"}
!CHECK:  omp.target map_entries(%[[MAP]] -> %[[ARG0:.*]] : !fir.ref<!fir.type<_QFmaptype_derived_explicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>) {
!CHECK:    ^bb0(%[[ARG0]]: !fir.ref<!fir.type<_QFmaptype_derived_explicitTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>):
subroutine mapType_derived_explicit
    type :: scalar_and_array
      real(4) :: real
      integer(4) :: array(10)
      integer(4) :: int
    end type scalar_and_array
    type(scalar_and_array) :: scalar_arr 
    
    !$omp target map(tofrom: scalar_arr)
       scalar_arr%int = 1
    !$omp end target
end subroutine mapType_derived_explicit

!CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.type<_QFmaptype_derived_explicit_single_memberTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}> {bindc_name = "scalar_arr", uniq_name = "_QFmaptype_derived_explicit_single_memberEscalar_arr"}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "_QFmaptype_derived_explicit_single_memberEscalar_arr"} : (!fir.ref<!fir.type<_QFmaptype_derived_explicit_single_memberTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>) -> (!fir.ref<!fir.type<_QFmaptype_derived_explicit_single_memberTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.ref<!fir.type<_QFmaptype_derived_explicit_single_memberTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>)
!CHECK: %[[MEMBER:.*]] = hlfir.designate %[[DECLARE]]#0{"array"}   shape %{{.*}} : (!fir.ref<!fir.type<_QFmaptype_derived_explicit_single_memberTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
!CHECK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
!CHECK: %[[MEMBER_MAP:.*]] = omp.map.info var_ptr(%[[MEMBER]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<10xi32>> {name = "scalar_arr%array"}
!CHECK: %[[PARENT_MAP:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.type<_QFmaptype_derived_explicit_single_memberTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.type<_QFmaptype_derived_explicit_single_memberTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>) map_clauses(tofrom) capture(ByRef) members(%[[MEMBER_MAP]] : [1] : !fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.type<_QFmaptype_derived_explicit_single_memberTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>> {name = "scalar_arr", partial_map = true}
!CHECK: omp.target map_entries(%[[MEMBER_MAP]] -> %[[ARG0:.*]], %[[PARENT_MAP]] -> %[[ARG1:.*]] : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.type<_QFmaptype_derived_explicit_single_memberTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>) {
!CHECK:  ^bb0(%[[ARG0]]: !fir.ref<!fir.array<10xi32>>, %[[ARG1]]: !fir.ref<!fir.type<_QFmaptype_derived_explicit_single_memberTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>):
subroutine mapType_derived_explicit_single_member
    type :: scalar_and_array
      real(4) :: real
      integer(4) :: array(10)
      integer(4) :: int
    end type scalar_and_array
    type(scalar_and_array) :: scalar_arr 
    
    !$omp target map(tofrom: scalar_arr%array)
       scalar_arr%array(1) = 1
    !$omp end target
end subroutine mapType_derived_explicit_single_member

!CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.type<_QFmaptype_derived_explicit_multiple_membersTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}> {bindc_name = "scalar_arr", uniq_name = "_QFmaptype_derived_explicit_multiple_membersEscalar_arr"}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "_QFmaptype_derived_explicit_multiple_membersEscalar_arr"} : (!fir.ref<!fir.type<_QFmaptype_derived_explicit_multiple_membersTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>) -> (!fir.ref<!fir.type<_QFmaptype_derived_explicit_multiple_membersTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.ref<!fir.type<_QFmaptype_derived_explicit_multiple_membersTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>)
!CHECK: %[[MEMBER1:.*]] = hlfir.designate %[[DECLARE]]#0{"int"}   : (!fir.ref<!fir.type<_QFmaptype_derived_explicit_multiple_membersTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>) -> !fir.ref<i32>
!CHECK: %[[MEMBER_MAP_1:.*]] = omp.map.info var_ptr(%[[MEMBER1]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "scalar_arr%int"}
!CHECK: %[[MEMBER2:.*]] = hlfir.designate %[[DECLARE]]#0{"real"}   : (!fir.ref<!fir.type<_QFmaptype_derived_explicit_multiple_membersTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>) -> !fir.ref<f32>
!CHECK: %[[MEMBER_MAP_2:.*]] = omp.map.info var_ptr(%[[MEMBER2]] : !fir.ref<f32>, f32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<f32> {name = "scalar_arr%real"}
!CHECK: %[[PARENT_MAP:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.type<_QFmaptype_derived_explicit_multiple_membersTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.type<_QFmaptype_derived_explicit_multiple_membersTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>) map_clauses(tofrom) capture(ByRef) members(%[[MEMBER_MAP_1]], %[[MEMBER_MAP_2]] : [2], [0] : !fir.ref<i32>, !fir.ref<f32>) -> !fir.ref<!fir.type<_QFmaptype_derived_explicit_multiple_membersTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>> {name = "scalar_arr", partial_map = true}
!CHECK: omp.target map_entries(%[[MEMBER_MAP_1]] -> %[[ARG0:.*]], %[[MEMBER_MAP_2]] -> %[[ARG1:.*]], %[[PARENT_MAP]] -> %[[ARG2:.*]] : !fir.ref<i32>, !fir.ref<f32>, !fir.ref<!fir.type<_QFmaptype_derived_explicit_multiple_membersTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>) {
!CHECK:  ^bb0(%[[ARG0]]: !fir.ref<i32>, %[[ARG1]]: !fir.ref<f32>, %[[ARG2]]: !fir.ref<!fir.type<_QFmaptype_derived_explicit_multiple_membersTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>):
subroutine mapType_derived_explicit_multiple_members
    type :: scalar_and_array
      real(4) :: real
      integer(4) :: array(10)
      integer(4) :: int
    end type scalar_and_array
    type(scalar_and_array) :: scalar_arr 
    
    !$omp target map(tofrom: scalar_arr%int, scalar_arr%real)
       scalar_arr%int = 1
    !$omp end target
end subroutine mapType_derived_explicit_multiple_members
  
!CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.type<_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}> {bindc_name = "scalar_arr", uniq_name = "_QFmaptype_derived_explicit_member_with_boundsEscalar_arr"}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "_QFmaptype_derived_explicit_member_with_boundsEscalar_arr"} : (!fir.ref<!fir.type<_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>) -> (!fir.ref<!fir.type<_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.ref<!fir.type<_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>)
!CHECK: %[[MEMBER:.*]] = hlfir.designate %[[DECLARE]]#0{"array"}   shape %{{.*}} : (!fir.ref<!fir.type<_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
!CHECK: %{{.*}} = arith.constant 1 : index
!CHECK: %[[LB:.*]] = arith.constant 1 : index
!CHECK: %[[UB:.*]] = arith.constant 4 : index
!CHECK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[UB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
!CHECK: %[[MEMBER_MAP:.*]] = omp.map.info var_ptr(%[[MEMBER]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%20) -> !fir.ref<!fir.array<10xi32>> {name = "scalar_arr%array(2:5)"}
!CHECK: %[[PARENT_MAP:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.type<_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.type<_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>) map_clauses(tofrom) capture(ByRef) members(%[[MEMBER_MAP]] : [1] : !fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.type<_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>> {name = "scalar_arr", partial_map = true}
!CHECK: omp.target map_entries(%[[MEMBER_MAP]] -> %[[ARG0:.*]], %[[PARENT_MAP]] -> %[[ARG1:.*]] : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.type<_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>) {
!CHECK: ^bb0(%[[ARG0]]: !fir.ref<!fir.array<10xi32>>, %[[ARG1]]: !fir.ref<!fir.type<_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array{real:f32,array:!fir.array<10xi32>,int:i32}>>):
subroutine mapType_derived_explicit_member_with_bounds
    type :: scalar_and_array
      real(4) :: real
      integer(4) :: array(10)
      integer(4) :: int
    end type scalar_and_array
    type(scalar_and_array) :: scalar_arr 
    
    !$omp target map(tofrom: scalar_arr%array(2:5))
       scalar_arr%array(3) = 3
    !$omp end target
end subroutine mapType_derived_explicit_member_with_bounds

!CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.type<_QFmaptype_derived_nested_explicit_single_memberTscalar_and_array{real:f32,array:!fir.array<10xi32>,nest:!fir.type<_QFmaptype_derived_nested_explicit_single_memberTnested{int:i32,real:f32,array:!fir.array<10xi32>}>,int:i32}> {bindc_name = "scalar_arr", uniq_name = "_QFmaptype_derived_nested_explicit_single_memberEscalar_arr"}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "_QFmaptype_derived_nested_explicit_single_memberEscalar_arr"} : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_single_memberTscalar_and_array{real:f32,array:!fir.array<10xi32>,nest:!fir.type<_QFmaptype_derived_nested_explicit_single_memberTnested{int:i32,real:f32,array:!fir.array<10xi32>}>,int:i32}>>) -> {{.*}}
!CHECK: %[[NEST:.*]] = hlfir.designate %[[DECLARE]]#0{"nest"}   : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_single_memberTscalar_and_array{real:f32,array:!fir.array<10xi32>,nest:!fir.type<_QFmaptype_derived_nested_explicit_single_memberTnested{int:i32,real:f32,array:!fir.array<10xi32>}>,int:i32}>>) -> {{.*}}
!CHECK: %[[NEST_MEMBER:.*]] = hlfir.designate %[[NEST]]{"array"}   shape %{{.*}} : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_single_memberTnested{int:i32,real:f32,array:!fir.array<10xi32>}>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
!CHECK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
!CHECK: %[[MEMBER_MAP:.*]] = omp.map.info var_ptr(%[[NEST_MEMBER]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<10xi32>> {name = "scalar_arr%nest%array"}
!CHECK: %[[PARENT_MAP:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : {{.*}}) map_clauses(tofrom) capture(ByRef) members(%35 : [2,2] : !fir.ref<!fir.array<10xi32>>) -> {{.*}} {name = "scalar_arr", partial_map = true}
!CHECK: omp.target map_entries(%[[MEMBER_MAP]] -> %[[ARG0:.*]], %[[PARENT_MAP]] -> %[[ARG1:.*]] : {{.*}}, {{.*}}) {
!CHECK:  ^bb0(%[[ARG0]]: {{.*}}, %[[ARG1]]: {{.*}}):
subroutine mapType_derived_nested_explicit_single_member
  type :: nested
    integer(4) :: int
    real(4) :: real
    integer(4) :: array(10)
  end type nested

  type :: scalar_and_array
    real(4) :: real
    integer(4) :: array(10)
    type(nested) :: nest
    integer(4) :: int
  end type scalar_and_array
  
  type(scalar_and_array) :: scalar_arr 

  !$omp target map(tofrom: scalar_arr%nest%array)
    scalar_arr%nest%array(1) = 1
  !$omp end target
end subroutine mapType_derived_nested_explicit_single_member

!CHECK: %[[ALLOCA:.*]] = fir.alloca {{.*}} {bindc_name = "scalar_arr", uniq_name = "_QFmaptype_derived_nested_explicit_multiple_membersEscalar_arr"}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "_QFmaptype_derived_nested_explicit_multiple_membersEscalar_arr"} : ({{.*}}) -> {{.*}}
!CHECK: %[[NEST:.*]] = hlfir.designate %[[DECLARE]]#0{"nest"}   : ({{.*}}) -> {{.*}}
!CHECK: %[[NEST_MEMBER1:.*]] = hlfir.designate %[[NEST]]{"int"}   : ({{.*}}) -> !fir.ref<i32>
!CHECK: %[[MEMBER_MAP_1:.*]] = omp.map.info var_ptr(%[[NEST_MEMBER1]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "scalar_arr%nest%int"}
!CHECK: %[[NEST:.*]] = hlfir.designate %[[DECLARE]]#0{"nest"}   : ({{.*}}) -> {{.*}}
!CHECK: %[[NEST_MEMBER2:.*]] = hlfir.designate %[[NEST]]{"real"}   : ({{.*}}) -> !fir.ref<f32>
!CHECK: %[[MEMBER_MAP_2:.*]] = omp.map.info var_ptr(%[[NEST_MEMBER2]] : !fir.ref<f32>, f32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<f32> {name = "scalar_arr%nest%real"}
!CHECK: %[[PARENT_MAP:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : {{.*}}, {{.*}}) map_clauses(tofrom) capture(ByRef) members(%[[MEMBER_MAP_1]], %[[MEMBER_MAP_2]] : [2,0], [2,1] : !fir.ref<i32>, !fir.ref<f32>) -> {{.*}} {name = "scalar_arr", partial_map = true}
!CHECK: omp.target map_entries(%[[MEMBER_MAP_1]] -> %[[ARG0:.*]], %[[MEMBER_MAP_2]] -> %[[ARG1:.*]], %[[PARENT_MAP]] -> %[[ARG2:.*]] : !fir.ref<i32>, !fir.ref<f32>, {{.*}}) {
!CHECK: ^bb0(%[[ARG0]]: !fir.ref<i32>, %[[ARG1]]: !fir.ref<f32>, %[[ARG2]]: {{.*}}):
subroutine mapType_derived_nested_explicit_multiple_members
  type :: nested
    integer(4) :: int
    real(4) :: real
    integer(4) :: array(10)
  end type nested

  type :: scalar_and_array
    real(4) :: real
    integer(4) :: array(10)
    type(nested) :: nest
    integer(4) :: int
  end type scalar_and_array

  type(scalar_and_array) :: scalar_arr 

  !$omp target map(tofrom: scalar_arr%nest%int, scalar_arr%nest%real)
    scalar_arr%nest%int = 1
  !$omp end target
end subroutine mapType_derived_nested_explicit_multiple_members

!CHECK: %[[ALLOCA:.*]] = fir.alloca {{.*}} {bindc_name = "scalar_arr", uniq_name = "_QFmaptype_derived_nested_explicit_member_with_boundsEscalar_arr"}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "_QFmaptype_derived_nested_explicit_member_with_boundsEscalar_arr"} : {{.*}} -> {{.*}}
!CHECK: %[[NEST:.*]] = hlfir.designate %[[DECLARE]]#0{"nest"}   : {{.*}} -> {{.*}}
!CHECK: %[[C10:.*]] = arith.constant 10 : index
!CHECK: %[[NEST_MEMBER:.*]] = hlfir.designate %[[NEST]]{"array"}   {{.*}} : {{.*}} -> !fir.ref<!fir.array<10xi32>>
!CHECK: %[[C1:.*]] = arith.constant 1 : index
!CHECK: %[[C1_2:.*]] = arith.constant 1 : index
!CHECK: %[[C4:.*]] = arith.constant 4 : index
!CHECK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[C1_2]] : index) upper_bound(%[[C4]] : index) extent(%[[C10]] : index) stride(%[[C1]] : index) start_idx(%[[C1]] : index)
!CHECK: %[[MEMBER_MAP:.*]] = omp.map.info var_ptr(%[[NEST_MEMBER]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<10xi32>> {name = "scalar_arr%nest%array(2:5)"}
!CHECK: %[[PARENT_MAP:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : {{.*}}, {{.*}}) map_clauses(tofrom) capture(ByRef) members(%[[MEMBER_MAP]] : [2,2] : !fir.ref<!fir.array<10xi32>>) -> {{.*}} {name = "scalar_arr", partial_map = true}
!CHECK: omp.target map_entries(%[[MEMBER_MAP]] -> %[[ARG0:.*]], %[[PARENT_MAP]] -> %[[ARG1:.*]] : !fir.ref<!fir.array<10xi32>>, {{.*}}) {
!CHECK: ^bb0(%[[ARG0]]: !fir.ref<!fir.array<10xi32>>, %[[ARG1]]: {{.*}}):
subroutine mapType_derived_nested_explicit_member_with_bounds
  type :: nested
    integer(4) :: int
    real(4) :: real
    integer(4) :: array(10)
  end type nested

  type :: scalar_and_array
    real(4) :: real
    integer(4) :: array(10)
    type(nested) :: nest
    integer(4) :: int
  end type scalar_and_array
  
  type(scalar_and_array) :: scalar_arr 
  
  !$omp target map(tofrom: scalar_arr%nest%array(2:5))
    scalar_arr%nest%array(3) = 3
  !$omp end target
end subroutine mapType_derived_nested_explicit_member_with_bounds

!CHECK: %[[ALLOCA_1:.*]] = fir.alloca {{.*}} {bindc_name = "scalar_arr1", uniq_name = "_QFmaptype_multilpe_derived_nested_explicit_memberEscalar_arr1"}
!CHECK: %[[DECLARE_1:.*]]:2 = hlfir.declare %[[ALLOCA_1]] {uniq_name = "_QFmaptype_multilpe_derived_nested_explicit_memberEscalar_arr1"} : {{.*}} -> {{.*}}
!CHECK: %[[ALLOCA_2:.*]] = fir.alloca {{.*}} {bindc_name = "scalar_arr2", uniq_name = "_QFmaptype_multilpe_derived_nested_explicit_memberEscalar_arr2"}
!CHECK: %[[DECLARE_2:.*]]:2 = hlfir.declare %[[ALLOCA_2]] {uniq_name = "_QFmaptype_multilpe_derived_nested_explicit_memberEscalar_arr2"} : {{.*}} -> {{.*}}
!CHECK: %[[PARENT_1:.*]] = hlfir.designate %[[DECLARE_1]]#0{"nest"}   : {{.*}} -> {{.*}}
!CHECK: %[[MEMBER_1:.*]] = hlfir.designate %[[PARENT_1]]{"int"}   : {{.*}} -> !fir.ref<i32>
!CHECK: %[[MAP_MEMBER_1:.*]] = omp.map.info var_ptr(%[[MEMBER_1]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "scalar_arr1%nest%int"}
!CHECK: %[[PARENT_2:.*]] = hlfir.designate %[[DECLARE_2]]#0{"nest"}   : {{.*}} -> {{.*}}
!CHECK: %[[MEMBER_2:.*]] = hlfir.designate %[[PARENT_2]]{"int"}   : {{.*}} -> !fir.ref<i32>
!CHECK: %[[MAP_MEMBER_2:.*]] = omp.map.info var_ptr(%[[MEMBER_2]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "scalar_arr2%nest%int"}
!CHECK: %[[MAP_PARENT_1:.*]] = omp.map.info var_ptr(%[[DECLARE_1]]#1 : {{.*}}) map_clauses(tofrom) capture(ByRef) members(%[[MAP_MEMBER_1]] : [2,0] : !fir.ref<i32>) -> {{.*}} {name = "scalar_arr1", partial_map = true}
!CHECK: %[[MAP_PARENT_2:.*]] = omp.map.info var_ptr(%[[DECLARE_2]]#1 : {{.*}}) map_clauses(tofrom) capture(ByRef) members(%[[MAP_MEMBER_2]] : [2,0] : !fir.ref<i32>) -> {{.*}} {name = "scalar_arr2", partial_map = true}
!CHECK: omp.target map_entries(%[[MAP_MEMBER_1]] -> %[[ARG0:.*]], %[[MAP_PARENT_1]] -> %[[ARG1:.*]], %[[MAP_MEMBER_2]] -> %[[ARG2:.*]], %[[MAP_PARENT_2:.*]] -> %[[ARG3:.*]] : !fir.ref<i32>, {{.*}}, !fir.ref<i32>, {{.*}}) {
!CHECK: ^bb0(%[[ARG0]]: !fir.ref<i32>, %[[ARG1]]: {{.*}}, %[[ARG2]]: !fir.ref<i32>, %[[ARG3]]: {{.*}}):
subroutine mapType_multilpe_derived_nested_explicit_member
  type :: nested
    integer(4) :: int
    real(4) :: real
    integer(4) :: array(10)
  end type nested

  type :: scalar_and_array
    real(4) :: real
    integer(4) :: array(10)
    type(nested) :: nest
    integer(4) :: int
  end type scalar_and_array
  
  type(scalar_and_array) :: scalar_arr1
  type(scalar_and_array) :: scalar_arr2

!$omp target map(tofrom:scalar_arr1%nest%int, scalar_arr2%nest%int)
  scalar_arr1%nest%int = 3
  scalar_arr2%nest%int = 2
!$omp end target
end subroutine mapType_multilpe_derived_nested_explicit_member
