! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s

! Tests for the iterator modifier on map and to/from motion clauses.

!===============================================================================
! target update
!===============================================================================

subroutine target_update_to_simple()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target update to(iterator(i = 1:n): a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_to_simple()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_to_simpleEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   fir.store %[[IV_I32]] to %[[IV_MEM:.*]] : !fir.ref<i32>
! CHECK:   %[[IV_DECL:.*]]:2 = hlfir.declare %[[IV_MEM]]
! CHECK:   %[[IV_LD:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_LD]] : (i32) -> i64
! CHECK:   %[[IV_IDX:.*]] = fir.convert %[[IV_I64]] : (i64) -> index
! CHECK:   %[[LB:.*]] = arith.subi %[[IV_IDX]], %{{.*}} : index
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[LB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_from_simple()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target update from(iterator(i = 1:n): a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_from_simple()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_from_simpleEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(from) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_assumed_shape(a, n)
  integer, intent(in) :: n
  real :: a(:)
  integer :: i

  !$omp target update to(iterator(i = 1:n): a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_assumed_shape
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 {uniq_name = "_QFtarget_update_assumed_shapeEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[DIMS:.*]]:3 = fir.box_dims %[[A]]#0, %{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%[[DIMS]]#1 : index) stride(%[[DIMS]]#2 : index) start_idx(%{{.*}} : index) {stride_in_bytes = true}
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.array<?xf32>>, f32) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_assumed_shape_2d(a, n, m)
  integer, intent(in) :: n, m
  real :: a(:, :)
  integer :: i, j

  !$omp target update to(iterator(i = 1:n, j = 1:m): a(i, j))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_assumed_shape_2d
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 {uniq_name = "_QFtarget_update_assumed_shape_2dEa"}
! CHECK: %[[M:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 3
! CHECK-SAME: uniq_name = "_QFtarget_update_assumed_shape_2dEm"
! CHECK: %[[N:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 2
! CHECK-SAME: uniq_name = "_QFtarget_update_assumed_shape_2dEn"
! CHECK: %[[N_LB_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[N_UB_I32:.*]] = fir.load %[[N]]#0 : !fir.ref<i32>
! CHECK: %[[N_LB:.*]] = fir.convert %[[N_LB_I32]] : (i32) -> index
! CHECK: %[[N_UB:.*]] = fir.convert %[[N_UB_I32]] : (i32) -> index
! CHECK: %[[N_STEP:.*]] = arith.constant 1 : index
! CHECK: %[[M_LB_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[M_UB_I32:.*]] = fir.load %[[M]]#0 : !fir.ref<i32>
! CHECK: %[[M_LB:.*]] = fir.convert %[[M_LB_I32]] : (i32) -> index
! CHECK: %[[M_UB:.*]] = fir.convert %[[M_UB_I32]] : (i32) -> index
! CHECK: %[[M_STEP:.*]] = arith.constant 1 : index
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV0:.*]]: index,
! CHECK-SAME: %[[IV1:.*]]: index) =
! CHECK-SAME: (%[[N_LB]] to %[[N_UB]] step %[[N_STEP]],
! CHECK-SAME: %[[M_LB]] to %[[M_UB]] step %[[M_STEP]]) {
! CHECK: %[[IV0_I32:.*]] = fir.convert %[[IV0]] : (index) -> i32
! CHECK: fir.store %[[IV0_I32]] to %[[IV0_ADDR:.*]] : !fir.ref<i32>
! CHECK: %[[IV0_DECL:.*]]:2 = hlfir.declare %[[IV0_ADDR]]
! CHECK: %[[IV1_I32:.*]] = fir.convert %[[IV1]] : (index) -> i32
! CHECK: fir.store %[[IV1_I32]] to %[[IV1_ADDR:.*]] : !fir.ref<i32>
! CHECK: %[[IV1_DECL:.*]]:2 = hlfir.declare %[[IV1_ADDR]]
! CHECK: %[[START0:.*]] = arith.constant 1 : index
! CHECK: %[[DIM0:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[A]]#0, %[[DIM0]]
! CHECK: %[[IV0_LOAD:.*]] = fir.load %[[IV0_DECL]]#0
! CHECK: %[[IV0_I64:.*]] = fir.convert %[[IV0_LOAD]] : (i32) -> i64
! CHECK: %[[IV0_INDEX:.*]] = fir.convert %[[IV0_I64]] : (i64) -> index
! CHECK: %[[INDEX0:.*]] = arith.subi %[[IV0_INDEX]], %[[START0]]
! CHECK: %[[BOUNDS0:.*]] = omp.map.bounds
! CHECK-SAME: lower_bound(%[[INDEX0]] : index)
! CHECK-SAME: upper_bound(%[[INDEX0]] : index)
! CHECK-SAME: extent(%[[DIMS0]]#1 : index)
! CHECK-SAME: stride(%[[DIMS0]]#2 : index)
! CHECK-SAME: start_idx(%[[START0]] : index)
! CHECK-SAME: {stride_in_bytes = true}
! CHECK: %[[START1:.*]] = arith.constant 1 : index
! CHECK: %[[DIM1:.*]] = arith.constant 1 : index
! CHECK: %[[DIMS1:.*]]:3 = fir.box_dims %[[A]]#0, %[[DIM1]]
! CHECK: %[[IV1_LOAD:.*]] = fir.load %[[IV1_DECL]]#0
! CHECK: %[[IV1_I64:.*]] = fir.convert %[[IV1_LOAD]] : (i32) -> i64
! CHECK: %[[IV1_INDEX:.*]] = fir.convert %[[IV1_I64]] : (i64) -> index
! CHECK: %[[INDEX1:.*]] = arith.subi %[[IV1_INDEX]], %[[START1]]
! CHECK: %[[BOUNDS1:.*]] = omp.map.bounds
! CHECK-SAME: lower_bound(%[[INDEX1]] : index)
! CHECK-SAME: upper_bound(%[[INDEX1]] : index)
! CHECK-SAME: extent(%[[DIMS1]]#1 : index)
! CHECK-SAME: stride(%[[DIMS1]]#2 : index)
! CHECK-SAME: start_idx(%[[START1]] : index)
! CHECK-SAME: {stride_in_bytes = true}
! CHECK: %[[BASE:.*]] = fir.box_addr %[[A]]#0
! CHECK: %[[MAP:.*]] = omp.map.info
! CHECK-SAME: var_ptr(%[[BASE]] : !fir.ref<!fir.array<?x?xf32>>, f32)
! CHECK-SAME: map_clauses(to)
! CHECK-SAME: capture(ByRef)
! CHECK-SAME: bounds(%[[BOUNDS0]], %[[BOUNDS1]])
! CHECK-SAME: -> !llvm.ptr
! CHECK-SAME: {name = ""}
! CHECK: omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_allocatable(a, n)
  integer, allocatable :: a(:)
  integer, intent(in) :: n
  integer :: i

  !$omp target update to(iterator(i = 1:n): a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_allocatable
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_update_allocatableEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOX:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:   %[[DIMS0:.*]]:3 = fir.box_dims %[[BOX]], %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:   %[[DIMS1:.*]]:3 = fir.box_dims %[[BOX]], %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%[[DIMS1]]#1 : index) stride(%[[DIMS1]]#2 : index) start_idx(%[[DIMS0]]#0 : index) {stride_in_bytes = true}
! CHECK:   %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.heap<!fir.array<?xi32>>, i32) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_pointer(a, n)
  integer, pointer :: a(:)
  integer, intent(in) :: n
  integer :: i

  !$omp target update to(iterator(i = 1:n): a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_pointer
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtarget_update_pointerEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOX:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:   %[[DIMS0:.*]]:3 = fir.box_dims %[[BOX]], %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:   %[[DIMS1:.*]]:3 = fir.box_dims %[[BOX]], %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%[[DIMS1]]#1 : index) stride(%[[DIMS1]]#2 : index) start_idx(%[[DIMS0]]#0 : index) {stride_in_bytes = true}
! CHECK:   %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.ptr<!fir.array<?xi32>>, i32) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_allocatable_2d(a, n, m)
  integer, allocatable :: a(:, :)
  integer, intent(in) :: n, m
  integer :: i, j

  !$omp target update to(iterator(i = 1:n, j = 1:m): a(i, j))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_allocatable_2d
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_update_allocatable_2dEa"}
! CHECK: %[[M:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 3
! CHECK-SAME: uniq_name = "_QFtarget_update_allocatable_2dEm"
! CHECK: %[[N:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 2
! CHECK-SAME: uniq_name = "_QFtarget_update_allocatable_2dEn"
! CHECK: %[[N_LB_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[N_UB_I32:.*]] = fir.load %[[N]]#0 : !fir.ref<i32>
! CHECK: %[[N_LB:.*]] = fir.convert %[[N_LB_I32]] : (i32) -> index
! CHECK: %[[N_UB:.*]] = fir.convert %[[N_UB_I32]] : (i32) -> index
! CHECK: %[[N_STEP:.*]] = arith.constant 1 : index
! CHECK: %[[M_LB_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[M_UB_I32:.*]] = fir.load %[[M]]#0 : !fir.ref<i32>
! CHECK: %[[M_LB:.*]] = fir.convert %[[M_LB_I32]] : (i32) -> index
! CHECK: %[[M_UB:.*]] = fir.convert %[[M_UB_I32]] : (i32) -> index
! CHECK: %[[M_STEP:.*]] = arith.constant 1 : index
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV0:.*]]: index,
! CHECK-SAME: %[[IV1:.*]]: index) =
! CHECK-SAME: (%[[N_LB]] to %[[N_UB]] step %[[N_STEP]],
! CHECK-SAME: %[[M_LB]] to %[[M_UB]] step %[[M_STEP]]) {
! CHECK: %[[IV0_I32:.*]] = fir.convert %[[IV0]] : (index) -> i32
! CHECK: fir.store %[[IV0_I32]] to %[[IV0_ADDR:.*]] : !fir.ref<i32>
! CHECK: %[[IV0_DECL:.*]]:2 = hlfir.declare %[[IV0_ADDR]]
! CHECK: %[[IV1_I32:.*]] = fir.convert %[[IV1]] : (index) -> i32
! CHECK: fir.store %[[IV1_I32]] to %[[IV1_ADDR:.*]] : !fir.ref<i32>
! CHECK: %[[IV1_DECL:.*]]:2 = hlfir.declare %[[IV1_ADDR]]
! CHECK: %[[BOX:.*]] = fir.load %[[A]]#0
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS0_LB:.*]]:3 = fir.box_dims %[[BOX]], %[[C0]]
! CHECK: %[[C0_EXTENT:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS0_EXTENT:.*]]:3 = fir.box_dims %[[BOX]], %[[C0_EXTENT]]
! CHECK: %[[IV0_LOAD:.*]] = fir.load %[[IV0_DECL]]#0
! CHECK: %[[IV0_I64:.*]] = fir.convert %[[IV0_LOAD]] : (i32) -> i64
! CHECK: %[[IV0_INDEX:.*]] = fir.convert %[[IV0_I64]] : (i64) -> index
! CHECK: %[[INDEX0:.*]] = arith.subi %[[IV0_INDEX]], %[[DIMS0_LB]]#0
! CHECK: %[[BOUNDS0:.*]] = omp.map.bounds
! CHECK-SAME: lower_bound(%[[INDEX0]] : index)
! CHECK-SAME: upper_bound(%[[INDEX0]] : index)
! CHECK-SAME: extent(%[[DIMS0_EXTENT]]#1 : index)
! CHECK-SAME: stride(%[[DIMS0_EXTENT]]#2 : index)
! CHECK-SAME: start_idx(%[[DIMS0_LB]]#0 : index)
! CHECK-SAME: {stride_in_bytes = true}
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[DIMS1_LB:.*]]:3 = fir.box_dims %[[BOX]], %[[C1]]
! CHECK: %[[C1_EXTENT:.*]] = arith.constant 1 : index
! CHECK: %[[DIMS1_EXTENT:.*]]:3 = fir.box_dims %[[BOX]], %[[C1_EXTENT]]
! CHECK: %[[IV1_LOAD:.*]] = fir.load %[[IV1_DECL]]#0
! CHECK: %[[IV1_I64:.*]] = fir.convert %[[IV1_LOAD]] : (i32) -> i64
! CHECK: %[[IV1_INDEX:.*]] = fir.convert %[[IV1_I64]] : (i64) -> index
! CHECK: %[[INDEX1:.*]] = arith.subi %[[IV1_INDEX]], %[[DIMS1_LB]]#0
! CHECK: %[[BOUNDS1:.*]] = omp.map.bounds
! CHECK-SAME: lower_bound(%[[INDEX1]] : index)
! CHECK-SAME: upper_bound(%[[INDEX1]] : index)
! CHECK-SAME: extent(%[[DIMS1_EXTENT]]#1 : index)
! CHECK-SAME: stride(%[[DIMS1_EXTENT]]#2 : index)
! CHECK-SAME: start_idx(%[[DIMS1_LB]]#0 : index)
! CHECK-SAME: {stride_in_bytes = true}
! CHECK: %[[BASE:.*]] = fir.box_addr %[[BOX]]
! CHECK: %[[MAP:.*]] = omp.map.info
! CHECK-SAME: var_ptr(%[[BASE]] : !fir.heap<!fir.array<?x?xi32>>, i32)
! CHECK-SAME: map_clauses(to)
! CHECK-SAME: capture(ByRef)
! CHECK-SAME: bounds(%[[BOUNDS0]], %[[BOUNDS1]])
! CHECK-SAME: -> !llvm.ptr
! CHECK-SAME: {name = ""}
! CHECK: omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_deferred_char(a, n)
  character(:), allocatable :: a(:)
  integer, intent(in) :: n
  integer :: i

  !$omp target update to(iterator(i = 1:n): a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_deferred_char
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_update_deferred_charEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOX:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index) {stride_in_bytes = true}
! CHECK:   %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.char<1,?>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_class_star(a, n)
  class(*), allocatable :: a(:)
  integer, intent(in) :: n
  integer :: i

  !$omp target update to(iterator(i = 1:n): a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_class_star
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_update_class_starEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOX:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index) {stride_in_bytes = true}
! CHECK:   %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.class<!fir.heap<!fir.array<?xnone>>>) -> !fir.heap<!fir.array<?xnone>>
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.heap<!fir.array<?xnone>>, none) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_to_section()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target update to(iterator(i = 1:n-1): a(i:i+1))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_to_section()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_to_sectionEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   fir.store %[[IV_I32]] to %[[IV_MEM:.*]] : !fir.ref<i32>
! CHECK:   %[[IV_DECL:.*]]:2 = hlfir.declare %[[IV_MEM]]
! CHECK:   %[[IV_LB_LD:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[LB_I64:.*]] = fir.convert %[[IV_LB_LD]] : (i32) -> i64
! CHECK:   %[[LB_IDX:.*]] = fir.convert %[[LB_I64]] : (i64) -> index
! CHECK:   %[[LB:.*]] = arith.subi %[[LB_IDX]], %{{.*}} : index
! CHECK:   %[[IV_UB_LD:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:   %[[UB_EXPR:.*]] = arith.addi %[[IV_UB_LD]], %[[C1_I32]] : i32
! CHECK:   %[[UB_I64:.*]] = fir.convert %[[UB_EXPR]] : (i32) -> i64
! CHECK:   %[[UB_IDX:.*]] = fir.convert %[[UB_I64]] : (i64) -> index
! CHECK:   %[[UB:.*]] = arith.subi %[[UB_IDX]], %{{.*}} : index
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[UB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_2d()
  integer, parameter :: n = 4, m = 6
  integer :: a(n, m)
  integer :: i, j

  !$omp target update to(iterator(i = 1:n, j = 1:m): a(i, j))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_2d()
! CHECK: %[[EXT_I:.*]] = arith.constant 4 : index
! CHECK: %[[EXT_J:.*]] = arith.constant 6 : index
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_2dEa"}
! CHECK: %[[I_LB_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[I_UB_I32:.*]] = arith.constant 4 : i32
! CHECK: %[[I_LB:.*]] = fir.convert %[[I_LB_I32]]
! CHECK-SAME: (i32) -> index
! CHECK: %[[I_UB:.*]] = fir.convert %[[I_UB_I32]]
! CHECK-SAME: (i32) -> index
! CHECK: %[[I_STEP:.*]] = arith.constant 1 : index
! CHECK: %[[J_LB_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[J_UB_I32:.*]] = arith.constant 6 : i32
! CHECK: %[[J_LB:.*]] = fir.convert %[[J_LB_I32]]
! CHECK-SAME: (i32) -> index
! CHECK: %[[J_UB:.*]] = fir.convert %[[J_UB_I32]]
! CHECK-SAME: (i32) -> index
! CHECK: %[[J_STEP:.*]] = arith.constant 1 : index
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV_I:.*]]: index,
! CHECK-SAME: %[[IV_J:.*]]: index) =
! CHECK-SAME: (%[[I_LB]] to %[[I_UB]] step %[[I_STEP]],
! CHECK-SAME: %[[J_LB]] to %[[J_UB]] step %[[J_STEP]]) {
! CHECK:   %[[IV_I_I32:.*]] = fir.convert %[[IV_I]]
! CHECK-SAME: (index) -> i32
! CHECK:   fir.store %[[IV_I_I32]] to %[[IV_I_MEM:.*]] : !fir.ref<i32>
! CHECK:   %[[IV_I_DECL:.*]]:2 = hlfir.declare %[[IV_I_MEM]]
! CHECK:   %[[IV_J_I32:.*]] = fir.convert %[[IV_J]]
! CHECK-SAME: (index) -> i32
! CHECK:   fir.store %[[IV_J_I32]] to %[[IV_J_MEM:.*]] : !fir.ref<i32>
! CHECK:   %[[IV_J_DECL:.*]]:2 = hlfir.declare %[[IV_J_MEM]]
! CHECK:   %[[IV_I_LD:.*]] = fir.load %[[IV_I_DECL]]#0
! CHECK-SAME: !fir.ref<i32>
! CHECK:   %[[IV_I_I64:.*]] = fir.convert %[[IV_I_LD]]
! CHECK-SAME: (i32) -> i64
! CHECK:   %[[IV_I_IDX:.*]] = fir.convert %[[IV_I_I64]]
! CHECK-SAME: (i64) -> index
! CHECK:   %[[LOC_I:.*]] = arith.subi %[[IV_I_IDX]], %{{.*}} : index
! CHECK:   %[[B_I:.*]] = omp.map.bounds
! CHECK-SAME: lower_bound(%[[LOC_I]] : index)
! CHECK-SAME: upper_bound(%[[LOC_I]] : index)
! CHECK-SAME: extent(%[[EXT_I]] : index)
! CHECK:   %[[IV_J_LD:.*]] = fir.load %[[IV_J_DECL]]#0
! CHECK-SAME: !fir.ref<i32>
! CHECK:   %[[IV_J_I64:.*]] = fir.convert %[[IV_J_LD]]
! CHECK-SAME: (i32) -> i64
! CHECK:   %[[IV_J_IDX:.*]] = fir.convert %[[IV_J_I64]]
! CHECK-SAME: (i64) -> index
! CHECK:   %[[LOC_J:.*]] = arith.subi %[[IV_J_IDX]], %{{.*}} : index
! CHECK:   %[[B_J:.*]] = omp.map.bounds
! CHECK-SAME: lower_bound(%[[LOC_J]] : index)
! CHECK-SAME: upper_bound(%[[LOC_J]] : index)
! CHECK-SAME: extent(%[[EXT_J]] : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0
! CHECK-SAME: : !fir.ref<!fir.array<4x6xi32>>,
! CHECK-SAME: !fir.array<4x6xi32>) map_clauses(to)
! CHECK-SAME: capture(ByRef) bounds(%[[B_I]], %[[B_J]])
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_step()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target update to(iterator(i = 1:n:2): a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_step()
! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
! CHECK: %[[LB:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
! CHECK: %[[UB:.*]] = fir.convert %[[C16_I32]] : (i32) -> index
! CHECK: %[[C2_I32:.*]] = arith.constant 2 : i32
! CHECK: %[[STEP:.*]] = fir.convert %[[C2_I32]] : (i32) -> index
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_negative_step()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target update to(iterator(i = n:1:-1): a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_negative_step()
! CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[LB:.*]] = fir.convert %[[C16_I32]] : (i32) -> index
! CHECK: %[[UB:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
! CHECK: %[[CM1_I32:.*]] = arith.constant -1 : i32
! CHECK: %[[STEP:.*]] = fir.convert %[[CM1_I32]] : (i32) -> index
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_multi_obj()
  integer, parameter :: n = 16
  integer :: a(n), b(n)
  integer :: i

  !$omp target update to(iterator(i = 1:n): a(i), b(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_multi_obj()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_multi_objEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_multi_objEb"}
! CHECK: %[[IT1:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS1:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP1:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS1]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP1]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT2:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS2:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP2:.*]] = omp.map.info var_ptr(%[[B]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS2]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP2]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT1]], %[[IT2]] : !omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>)

subroutine target_update_mixed_same_clause()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target update to(iterator(i = 2:n:2): a(1), a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_mixed_same_clause()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_mixed_same_clauseEa"}
! CHECK: %[[MAP_PLAIN:.*]] = omp.map.info var_ptr(%[[A]]#1 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "a(1)"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS_IT:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP_IT:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS_IT]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP_IT]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_entries(%[[MAP_PLAIN]] : !fir.ref<!fir.array<16xi32>>) map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_multi_clause()
  integer, parameter :: n = 8
  integer :: a(n), b(n)
  integer :: i, j

  !$omp target update to(iterator(i = 1:n): a(i)) &
  !$omp&              from(iterator(j = 1:n:2): b(j))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_multi_clause()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_multi_clauseEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_multi_clauseEb"}
! CHECK: %[[IT1:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS1:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP1:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<8xi32>>, !fir.array<8xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS1]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP1]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT2:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS2:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP2:.*]] = omp.map.info var_ptr(%[[B]]#0 : !fir.ref<!fir.array<8xi32>>, !fir.array<8xi32>) map_clauses(from) capture(ByRef) bounds(%[[BOUNDS2]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP2]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT1]], %[[IT2]] : !omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>)

subroutine target_update_mixed_clauses()
  integer, parameter :: n = 16
  integer :: a(n), b(n)
  integer :: i

  !$omp target update to(iterator(i = 1:n): a(i)) from(b)
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_mixed_clauses()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_mixed_clausesEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_mixed_clausesEb"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS_IT:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP_IT:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS_IT]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP_IT]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP_B:.*]] = omp.map.info var_ptr(%[[B]]#1 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(from) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "b"}
! CHECK: omp.target_update map_entries(%[[MAP_B]] : !fir.ref<!fir.array<16xi32>>) map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_mapper()
  type :: s
    integer :: a
  end type
  type(s) :: x(10)
  integer :: i

  !$omp declare mapper(m: s :: v) map(to: v%a)
  !$omp target update to(mapper(m), iterator(i = 1:10): x(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_mapper()
! CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_mapperEx"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[X]]#0 : !fir.ref<!fir.array<10x!fir.type<_QFtarget_update_mapperTs{a:i32}>>>, !fir.array<10x!fir.type<_QFtarget_update_mapperTs{a:i32}>>) map_clauses(to) capture(ByRef) mapper(@_QQFtarget_update_mapperm) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_default_mapper()
  type :: s
    integer :: a
  end type
  type(s) :: x

  !$omp declare mapper(s :: v) map(to: v%a)
  !$omp target update to(x)
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_default_mapper()
! CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtarget_update_default_mapperEx"}
! CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[X]]#1 : !fir.ref<!fir.type<_QFtarget_update_default_mapperTs{a:i32}>>, !fir.type<_QFtarget_update_default_mapperTs{a:i32}>) map_clauses(to) capture(ByRef) mapper(@_QQFtarget_update_default_mappers_omp_default_mapper) -> !fir.ref<!fir.type<_QFtarget_update_default_mapperTs{a:i32}>> {name = "x"}
! CHECK: omp.target_update map_entries(%[[MAP]] : !fir.ref<!fir.type<_QFtarget_update_default_mapperTs{a:i32}>>)

subroutine target_update_iterated_default_mapper()
  type :: s
    integer :: a
  end type
  type(s) :: x(10)
  integer :: i

  !$omp declare mapper(s :: v) map(to: v%a)
  !$omp target update to(iterator(i = 1:10): x(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_iterated_default_mapper()
! CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_iterated_default_mapperEx"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[X]]#0 : !fir.ref<!fir.array<10x!fir.type<_QFtarget_update_iterated_default_mapperTs{a:i32}>>>, !fir.array<10x!fir.type<_QFtarget_update_iterated_default_mapperTs{a:i32}>>) map_clauses(to) capture(ByRef) mapper(@_QQFtarget_update_iterated_default_mappers_omp_default_mapper) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! Assumed-shape array of a derived type with a mapper, mapped per iteration.
subroutine target_update_assumed_shape_mapper(x, n)
  type :: s
    integer :: a
  end type
  type(s) :: x(:)
  integer, intent(in) :: n
  integer :: i

  !$omp declare mapper(s :: v) map(to: v%a)
  !$omp target update to(iterator(i = 1:n): x(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_assumed_shape_mapper
! CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 {uniq_name = "_QFtarget_update_assumed_shape_mapperEx"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[DIMS:.*]]:3 = fir.box_dims %[[X]]#0, %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QFtarget_update_assumed_shape_mapperTs{a:i32}>>>, index) -> (index, index, index)
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%[[DIMS]]#1 : index) stride(%[[DIMS]]#2 : index) start_idx(%{{.*}} : index) {stride_in_bytes = true}
! CHECK:   %[[BASE:.*]] = fir.box_addr %[[X]]#0 : (!fir.box<!fir.array<?x!fir.type<_QFtarget_update_assumed_shape_mapperTs{a:i32}>>>) -> !fir.ref<!fir.array<?x!fir.type<_QFtarget_update_assumed_shape_mapperTs{a:i32}>>>
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.ref<!fir.array<?x!fir.type<_QFtarget_update_assumed_shape_mapperTs{a:i32}>>>, !fir.type<_QFtarget_update_assumed_shape_mapperTs{a:i32}>) map_clauses(to) capture(ByRef) mapper(@_QQFtarget_update_assumed_shape_mappers_omp_default_mapper) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

!===============================================================================
! target data
!===============================================================================

subroutine target_data_section()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target data map(iterator(i = 1:n-1), tofrom: a(i:i+1))
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPtarget_data_section()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_data_sectionEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   fir.store %[[IV_I32]] to %[[IV_MEM:.*]] : !fir.ref<i32>
! CHECK:   %[[IV_DECL:.*]]:2 = hlfir.declare %[[IV_MEM]]
! CHECK:   %[[IV_LB_LD:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[LB_I64:.*]] = fir.convert %[[IV_LB_LD]] : (i32) -> i64
! CHECK:   %[[LB_IDX:.*]] = fir.convert %[[LB_I64]] : (i64) -> index
! CHECK:   %[[LB:.*]] = arith.subi %[[LB_IDX]], %{{.*}} : index
! CHECK:   %[[IV_UB_LD:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:   %[[UB_EXPR:.*]] = arith.addi %[[IV_UB_LD]], %[[C1_I32]] : i32
! CHECK:   %[[UB_I64:.*]] = fir.convert %[[UB_EXPR]] : (i32) -> i64
! CHECK:   %[[UB_IDX:.*]] = fir.convert %[[UB_I64]] : (i64) -> index
! CHECK:   %[[UB:.*]] = arith.subi %[[UB_IDX]], %{{.*}} : index
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[UB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_data map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_data_mapper()
  type :: s
    integer :: a
  end type
  type(s) :: x(10)
  integer :: i

  !$omp declare mapper(m: s :: v) map(to: v%a)
  !$omp target data map(mapper(m), iterator(i = 1:10), tofrom: x(i))
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPtarget_data_mapper()
! CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_data_mapperEx"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[X]]#0 : !fir.ref<!fir.array<10x!fir.type<_QFtarget_data_mapperTs{a:i32}>>>, !fir.array<10x!fir.type<_QFtarget_data_mapperTs{a:i32}>>) map_clauses(tofrom) capture(ByRef) mapper(@_QQFtarget_data_mapperm) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_data map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_data_default_mapper()
  type :: s
    integer :: a
  end type
  type(s) :: x(10)
  integer :: i

  !$omp declare mapper(s :: v) map(to: v%a)
  !$omp target data map(iterator(i = 1:10), tofrom: x(i))
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPtarget_data_default_mapper()
! CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_data_default_mapperEx"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[X]]#0 : !fir.ref<!fir.array<10x!fir.type<_QFtarget_data_default_mapperTs{a:i32}>>>, !fir.array<10x!fir.type<_QFtarget_data_default_mapperTs{a:i32}>>) map_clauses(tofrom) capture(ByRef) mapper(@_QQFtarget_data_default_mappers_omp_default_mapper) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_data map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_data_implicit_mapper()
  type :: s
    integer, allocatable :: a(:)
  end type
  type(s) :: x(10)
  integer :: i

  !$omp target data map(iterator(i = 1:10), tofrom: x(i))
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPtarget_data_implicit_mapper()
! CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_data_implicit_mapperEx"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[X]]#0 : !fir.ref<!fir.array<10x!fir.type<_QFtarget_data_implicit_mapperTs{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>, !fir.array<10x!fir.type<_QFtarget_data_implicit_mapperTs{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) map_clauses(tofrom) capture(ByRef) mapper(@{{.*omp_default_mapper}}) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_data map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

!===============================================================================
! target enter data
!===============================================================================

subroutine target_enter_data_simple()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target enter data map(iterator(i = 1:n), to: a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_enter_data_simple()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_enter_data_simpleEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   fir.store %[[IV_I32]] to %[[IV_MEM:.*]] : !fir.ref<i32>
! CHECK:   %[[IV_DECL:.*]]:2 = hlfir.declare %[[IV_MEM]]
! CHECK:   %[[IV_LD:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_LD]] : (i32) -> i64
! CHECK:   %[[IV_IDX:.*]] = fir.convert %[[IV_I64]] : (i64) -> index
! CHECK:   %[[LB:.*]] = arith.subi %[[IV_IDX]], %{{.*}} : index
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[LB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_enter_data map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_enter_data_section()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target enter data map(iterator(i = 1:n-2), to: a(i:i+2))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_enter_data_section()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_enter_data_sectionEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   fir.store %[[IV_I32]] to %[[IV_MEM:.*]] : !fir.ref<i32>
! CHECK:   %[[IV_DECL:.*]]:2 = hlfir.declare %[[IV_MEM]]
! CHECK:   %[[IV_LB_LD:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[LB_I64:.*]] = fir.convert %[[IV_LB_LD]] : (i32) -> i64
! CHECK:   %[[LB_IDX:.*]] = fir.convert %[[LB_I64]] : (i64) -> index
! CHECK:   %[[LB:.*]] = arith.subi %[[LB_IDX]], %{{.*}} : index
! CHECK:   %[[IV_UB_LD:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[C2_I32:.*]] = arith.constant 2 : i32
! CHECK:   %[[UB_EXPR:.*]] = arith.addi %[[IV_UB_LD]], %[[C2_I32]] : i32
! CHECK:   %[[UB_I64:.*]] = fir.convert %[[UB_EXPR]] : (i32) -> i64
! CHECK:   %[[UB_IDX:.*]] = fir.convert %[[UB_I64]] : (i64) -> index
! CHECK:   %[[UB:.*]] = arith.subi %[[UB_IDX]], %{{.*}} : index
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[UB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_enter_data map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! Expression-based subscript using multiple iterator variables: a((i-1)*m+j)
! maps a 2D logical iteration space onto a 1D array.
subroutine target_enter_data_expr_subscript()
  integer, parameter :: m = 4
  integer, parameter :: n = m * m
  integer :: a(n)
  integer :: i, j

  !$omp target enter data map(iterator(i = 1:m, j = 1:m), to: a((i-1)*m+j))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_enter_data_expr_subscript()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_enter_data_expr_subscriptEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV0:.*]]: index, %[[IV1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV0_I32:.*]] = fir.convert %[[IV0]] : (index) -> i32
! CHECK:   fir.store %[[IV0_I32]] to %[[IV0_MEM:.*]] : !fir.ref<i32>
! CHECK:   %[[IV0_DECL:.*]]:2 = hlfir.declare %[[IV0_MEM]]
! CHECK:   %[[IV1_I32:.*]] = fir.convert %[[IV1]] : (index) -> i32
! CHECK:   fir.store %[[IV1_I32]] to %[[IV1_MEM:.*]] : !fir.ref<i32>
! CHECK:   %[[IV1_DECL:.*]]:2 = hlfir.declare %[[IV1_MEM]]
! CHECK:   %[[IV0_LD:.*]] = fir.load %[[IV0_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:   %[[SUB:.*]] = arith.subi %[[IV0_LD]], %[[C1_I32]] : i32
! CHECK:   %[[NOREASSOC:.*]] = hlfir.no_reassoc %[[SUB]] : i32
! CHECK:   %[[MUL:.*]] = arith.muli %{{.*}}, %[[NOREASSOC]] : i32
! CHECK:   %[[IV1_LD:.*]] = fir.load %[[IV1_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[ADD:.*]] = arith.addi %[[MUL]], %[[IV1_LD]] : i32
! CHECK:   %[[IDX:.*]] = fir.convert %[[ADD]] : (i32) -> i64
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_enter_data map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

!===============================================================================
! target exit data
!===============================================================================

subroutine target_exit_data_simple()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target exit data map(iterator(i = 1:n), from: a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_exit_data_simple()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_exit_data_simpleEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   fir.store %[[IV_I32]] to %[[IV_MEM:.*]] : !fir.ref<i32>
! CHECK:   %[[IV_DECL:.*]]:2 = hlfir.declare %[[IV_MEM]]
! CHECK:   %[[IV_LD:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_LD]] : (i32) -> i64
! CHECK:   %[[IV_IDX:.*]] = fir.convert %[[IV_I64]] : (i64) -> index
! CHECK:   %[[LB:.*]] = arith.subi %[[IV_IDX]], %{{.*}} : index
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[LB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(from) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_exit_data map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! Multiple objects with negative step, producing separate iterators.
subroutine target_exit_data_multi_obj()
  integer, parameter :: n = 16
  integer :: a(n), b(n)
  integer :: i

  !$omp target exit data map(iterator(i = n:1:-1), from: a(i), b(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_exit_data_multi_obj()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_exit_data_multi_objEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_exit_data_multi_objEb"}
! CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[LB:.*]] = fir.convert %[[C16_I32]] : (i32) -> index
! CHECK: %[[UB:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
! CHECK: %[[CM1_I32:.*]] = arith.constant -1 : i32
! CHECK: %[[STEP:.*]] = fir.convert %[[CM1_I32]] : (i32) -> index
! CHECK: %[[IT1:.*]] = omp.iterator(%{{.*}}: index) = (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:   %[[BOUNDS1:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP1:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(from) capture(ByRef) bounds(%[[BOUNDS1]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP1]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT2:.*]] = omp.iterator(%{{.*}}: index) = (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:   %[[BOUNDS2:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP2:.*]] = omp.map.info var_ptr(%[[B]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(from) capture(ByRef) bounds(%[[BOUNDS2]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP2]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_exit_data map_iterated(%[[IT1]], %[[IT2]] : !omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>)
