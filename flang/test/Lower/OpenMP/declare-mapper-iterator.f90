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
! CHECK:     %[[A:.*]] = hlfir.designate %[[DECL]]#0{"a"} {{.*}} : (!fir.ref<!fir.type<_QFdeclare_mapper_iteratorTs{{.*}}>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:     %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
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
! CHECK:   %[[IT:.*]] = omp.iterator(%{{.*}}: index, %{{.*}}: index) = ({{.*}}) {
! CHECK:     %[[A:.*]] = hlfir.designate %[[DECL]]#0{"a"}{{.*}} -> !fir.ref<!fir.array<4x6xi32>>
! CHECK:     %[[B0:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:     %[[B1:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[A]] : !fir.ref<!fir.array<4x6xi32>>, !fir.array<4x6xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[B0]], %[[B1]]) -> !llvm.ptr {name = ""}
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
! CHECK:   %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}}) {
! CHECK:     %[[BOX:.*]] = hlfir.designate %[[DECL]]#0{"a"}{{.*}} -> !fir.box<!fir.array<10xi32>>
! CHECK:     %[[DIMS:.*]]:3 = fir.box_dims %[[BOX]], %{{.*}} : (!fir.box<!fir.array<10xi32>>, index) -> (index, index, index)
! CHECK:     %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%[[DIMS]]#1 : index) stride(%[[DIMS]]#2 : index) start_idx(%{{.*}} : index) {stride_in_bytes = true}
! CHECK:     %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   } -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)
