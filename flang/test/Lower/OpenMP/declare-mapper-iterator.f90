! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s

subroutine declare_mapper_nondefault_lb()
  type :: t
    integer :: a(-2:7)
  end type
  !$omp declare mapper(m: t :: v) map(iterator(i = -2:6): v%a(i))
end

subroutine declare_mapper_alloc_section()
  type :: t
    integer, allocatable :: a(:)
  end type
  !$omp declare mapper(m: t :: v) map(iterator(i = 1:9): v%a(i:i+1))
end

subroutine declare_mapper_multi()
  type :: t
    integer :: a(10)
    integer :: b(10)
  end type
  !$omp declare mapper(m: t :: v) map(iterator(i = 1:10): v%a(i), v%b(i))
end

subroutine declare_mapper_section()
  type :: t
    integer :: a(10)
  end type
  !$omp declare mapper(m: t :: v) map(iterator(i = 1:9): v%a(i:i+1))
end

subroutine declare_mapper_2d()
  type :: t
    integer :: a(4, 6)
  end type
  !$omp declare mapper(m: t :: v) map(iterator(i = 1:4, j = 1:6): v%a(i, j))
end

subroutine declare_mapper_pointer()
  type :: t
    integer, pointer :: a(:)
  end type
  !$omp declare mapper(m: t :: v) map(iterator(i = 1:10): v%a(i))
end

subroutine declare_mapper_allocatable(arg)
  type :: t
    integer, allocatable :: a(:)
  end type
  type(t) :: arg(:)

  !$omp declare mapper(m: t :: v) map(iterator(i = 1:10): v%a(i))
end

subroutine declare_mapper_iterator(arg)
  type :: s
    integer :: a(10)
  end type
  type(s) :: arg(:)

  !$omp declare mapper(m: s :: v) map(iterator(i = 1:10): v%a(i))
end

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_iteratorm
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_iteratorTs{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] {uniq_name = "_QFdeclare_mapper_iteratorEv"}
! CHECK:   %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:     %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:     fir.store %[[IV_I32]] to %[[IV_MEM:.*]] : !fir.ref<i32>
! CHECK:     %[[IV_DECL:.*]]:2 = hlfir.declare %[[IV_MEM]]
! CHECK:     %[[A:.*]] = hlfir.designate %[[DECL]]#0{"a"} {{.*}} : (!fir.ref<!fir.type<_QFdeclare_mapper_iteratorTs{{.*}}>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:     %[[IV_LD:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
! CHECK:     %[[IV_I64:.*]] = fir.convert %[[IV_LD]] : (i32) -> i64
! CHECK:     %[[IV_IDX:.*]] = fir.convert %[[IV_I64]] : (i64) -> index
! CHECK:     %[[LB:.*]] = arith.subi %[[IV_IDX]], %{{.*}} : index
! CHECK:     %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[LB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[A]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:     omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK:   } -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_allocatablem
! CHECK: ^bb0(%[[ARG2:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_allocatableTt{{.*}}>):
! CHECK:   %[[DECL2:.*]]:2 = hlfir.declare %[[ARG2]] {uniq_name = "_QFdeclare_mapper_allocatableEv"}
! CHECK:   %[[IT2:.*]] = omp.iterator(%[[IV2:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:     %[[BOX_REF:.*]] = hlfir.designate %[[DECL2]]#0{"a"}{{.*}} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:     %[[BOX:.*]] = fir.load %[[BOX_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:     %[[DIMS0:.*]]:3 = fir.box_dims %[[BOX]], %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:     %[[DIMS1:.*]]:3 = fir.box_dims %[[BOX]], %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:     %[[BOUNDS2:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%[[DIMS1]]#1 : index) stride(%[[DIMS1]]#2 : index) start_idx(%[[DIMS0]]#0 : index) {stride_in_bytes = true}
! CHECK:     %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:     %[[MAP2:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.heap<!fir.array<?xi32>>, i32) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS2]]) -> !llvm.ptr {name = ""}
! CHECK:     omp.yield(%[[MAP2]] : !llvm.ptr)
! CHECK:   } -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT2]] : !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_pointerm
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_pointerTt{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] {uniq_name = "_QFdeclare_mapper_pointerEv"}
! CHECK:   %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}}) {
! CHECK:     %[[BOX_REF:.*]] = hlfir.designate %[[DECL]]#0{"a"}{{.*}} -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:     %[[BOX:.*]] = fir.load %[[BOX_REF]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:     %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.ptr<!fir.array<?xi32>>, i32) map_clauses(tofrom) capture(ByRef) bounds(%{{.*}}) -> !llvm.ptr {name = ""}
! CHECK:   } -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_2dm
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_2dTt{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] {uniq_name = "_QFdeclare_mapper_2dEv"}
! CHECK:   %[[I_LB_I32:.*]] = arith.constant 1 : i32
! CHECK:   %[[I_UB_I32:.*]] = arith.constant 4 : i32
! CHECK:   %[[I_LB:.*]] = fir.convert %[[I_LB_I32]]
! CHECK-SAME: (i32) -> index
! CHECK:   %[[I_UB:.*]] = fir.convert %[[I_UB_I32]]
! CHECK-SAME: (i32) -> index
! CHECK:   %[[I_STEP:.*]] = arith.constant 1 : index
! CHECK:   %[[J_LB_I32:.*]] = arith.constant 1 : i32
! CHECK:   %[[J_UB_I32:.*]] = arith.constant 6 : i32
! CHECK:   %[[J_LB:.*]] = fir.convert %[[J_LB_I32]]
! CHECK-SAME: (i32) -> index
! CHECK:   %[[J_UB:.*]] = fir.convert %[[J_UB_I32]]
! CHECK-SAME: (i32) -> index
! CHECK:   %[[J_STEP:.*]] = arith.constant 1 : index
! CHECK:   %[[IT:.*]] = omp.iterator(%[[IV_I:.*]]: index,
! CHECK-SAME: %[[IV_J:.*]]: index) =
! CHECK-SAME: (%[[I_LB]] to %[[I_UB]] step %[[I_STEP]],
! CHECK-SAME: %[[J_LB]] to %[[J_UB]] step %[[J_STEP]]) {
! CHECK:     %[[IV_I_I32:.*]] = fir.convert %[[IV_I]]
! CHECK-SAME: (index) -> i32
! CHECK:     fir.store %[[IV_I_I32]] to %[[IV_I_MEM:.*]] : !fir.ref<i32>
! CHECK:     %[[IV_I_DECL:.*]]:2 = hlfir.declare %[[IV_I_MEM]]
! CHECK:     %[[IV_J_I32:.*]] = fir.convert %[[IV_J]]
! CHECK-SAME: (index) -> i32
! CHECK:     fir.store %[[IV_J_I32]] to %[[IV_J_MEM:.*]] : !fir.ref<i32>
! CHECK:     %[[IV_J_DECL:.*]]:2 = hlfir.declare %[[IV_J_MEM]]
! CHECK:     %[[EXT_I:.*]] = arith.constant 4 : index
! CHECK:     %[[EXT_J:.*]] = arith.constant 6 : index
! CHECK:     %[[A:.*]] = hlfir.designate %[[DECL]]#0{"a"}{{.*}} -> !fir.ref<!fir.array<4x6xi32>>
! CHECK:     %[[IV_I_LD:.*]] = fir.load %[[IV_I_DECL]]#0
! CHECK-SAME: !fir.ref<i32>
! CHECK:     %[[IV_I_I64:.*]] = fir.convert %[[IV_I_LD]]
! CHECK-SAME: (i32) -> i64
! CHECK:     %[[IV_I_IDX:.*]] = fir.convert %[[IV_I_I64]]
! CHECK-SAME: (i64) -> index
! CHECK:     %[[LOC_I:.*]] = arith.subi %[[IV_I_IDX]], %{{.*}} : index
! CHECK:     %[[B_I:.*]] = omp.map.bounds
! CHECK-SAME: lower_bound(%[[LOC_I]] : index)
! CHECK-SAME: upper_bound(%[[LOC_I]] : index)
! CHECK-SAME: extent(%[[EXT_I]] : index)
! CHECK:     %[[IV_J_LD:.*]] = fir.load %[[IV_J_DECL]]#0
! CHECK-SAME: !fir.ref<i32>
! CHECK:     %[[IV_J_I64:.*]] = fir.convert %[[IV_J_LD]]
! CHECK-SAME: (i32) -> i64
! CHECK:     %[[IV_J_IDX:.*]] = fir.convert %[[IV_J_I64]]
! CHECK-SAME: (i64) -> index
! CHECK:     %[[LOC_J:.*]] = arith.subi %[[IV_J_IDX]], %{{.*}} : index
! CHECK:     %[[B_J:.*]] = omp.map.bounds
! CHECK-SAME: lower_bound(%[[LOC_J]] : index)
! CHECK-SAME: upper_bound(%[[LOC_J]] : index)
! CHECK-SAME: extent(%[[EXT_J]] : index)
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]
! CHECK-SAME: : !fir.ref<!fir.array<4x6xi32>>,
! CHECK-SAME: !fir.array<4x6xi32>) map_clauses(tofrom)
! CHECK-SAME: capture(ByRef) bounds(%[[B_I]], %[[B_J]])
! CHECK:   } -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_sectionm
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_sectionTt{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] {uniq_name = "_QFdeclare_mapper_sectionEv"}
! CHECK:   %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}}) {
! CHECK:     %[[A:.*]] = hlfir.designate %[[DECL]]#0{"a"}{{.*}} -> !fir.ref<!fir.array<10xi32>>
! CHECK:     %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
! CHECK:     %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[A]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   } -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_multim
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_multiTt{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] {uniq_name = "_QFdeclare_mapper_multiEv"}
! CHECK:   %[[IT_A:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}}) {
! CHECK:     %[[A:.*]] = hlfir.designate %[[DECL]]#0{"a"}{{.*}} -> !fir.ref<!fir.array<10xi32>>
! CHECK:     %[[MAP_A:.*]] = omp.map.info var_ptr(%[[A]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%{{.*}}) -> !llvm.ptr {name = ""}
! CHECK:   } -> !omp.iterated<!llvm.ptr>
! CHECK:   %[[IT_B:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}}) {
! CHECK:     %[[B:.*]] = hlfir.designate %[[DECL]]#0{"b"}{{.*}} -> !fir.ref<!fir.array<10xi32>>
! CHECK:     %[[MAP_B:.*]] = omp.map.info var_ptr(%[[B]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%{{.*}}) -> !llvm.ptr {name = ""}
! CHECK:   } -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT_A]], %[[IT_B]] : !omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_alloc_sectionm
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_alloc_sectionTt{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] {uniq_name = "_QFdeclare_mapper_alloc_sectionEv"}
! CHECK:   %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}}) {
! CHECK:     %[[BOX_REF:.*]] = hlfir.designate %[[DECL]]#0{"a"}{{.*}} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:     %[[BOX:.*]] = fir.load %[[BOX_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:     %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
! CHECK:     %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index) {stride_in_bytes = true}
! CHECK:     %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.heap<!fir.array<?xi32>>, i32) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   } -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_nondefault_lbm
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_nondefault_lbTt{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] {uniq_name = "_QFdeclare_mapper_nondefault_lbEv"}
! CHECK:   %[[LB_I32:.*]] = arith.constant -2 : i32
! CHECK:   %[[UB_I32:.*]] = arith.constant 6 : i32
! CHECK:   %[[LB:.*]] = fir.convert %[[LB_I32]] : (i32) -> index
! CHECK:   %[[UB:.*]] = fir.convert %[[UB_I32]] : (i32) -> index
! CHECK:   %[[STEP:.*]] = arith.constant 1 : index
! CHECK:   %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) =
! CHECK-SAME: (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:     %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:     fir.store %[[IV_I32]] to %[[IV_MEM:.*]] : !fir.ref<i32>
! CHECK:     %[[IV_DECL:.*]]:2 = hlfir.declare %[[IV_MEM]]
! CHECK:     %[[EXTENT:.*]] = arith.constant 10 : index
! CHECK:     %[[START:.*]] = arith.constant -2 : index
! CHECK:     %[[SHAPE:.*]] = fir.shape_shift %[[START]], %[[EXTENT]]
! CHECK:     %[[BOX:.*]] = hlfir.designate %[[DECL]]#0{"a"}
! CHECK-SAME: shape %[[SHAPE]]
! CHECK-SAME: -> !fir.box<!fir.array<10xi32>>
! CHECK:     %[[DIM:.*]] = arith.constant 0 : index
! CHECK:     %[[DIMS:.*]]:3 = fir.box_dims %[[BOX]], %[[DIM]]
! CHECK:     %[[IV_LD:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
! CHECK:     %[[IV_I64:.*]] = fir.convert %[[IV_LD]] : (i32) -> i64
! CHECK:     %[[IV_IDX:.*]] = fir.convert %[[IV_I64]] : (i64) -> index
! CHECK:     %[[OFFSET:.*]] = arith.subi %[[IV_IDX]], %[[START]] : index
! CHECK:     %[[BOUNDS:.*]] = omp.map.bounds
! CHECK-SAME: lower_bound(%[[OFFSET]] : index)
! CHECK-SAME: upper_bound(%[[OFFSET]] : index)
! CHECK-SAME: extent(%[[DIMS]]#1 : index)
! CHECK-SAME: stride(%[[DIMS]]#2 : index)
! CHECK-SAME: start_idx(%[[START]] : index)
! CHECK-SAME: {stride_in_bytes = true}
! CHECK:     %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   } -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)
