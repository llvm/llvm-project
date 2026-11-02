! RUN: rm -rf %t && split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   %t/maps.f90 -o - | FileCheck %t/maps.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   %t/ranges.f90 -o %t/ranges.mlir
! RUN: FileCheck %t/ranges.f90 < %t/ranges.mlir
! RUN: fir-opt --canonicalize --verify-each %t/ranges.mlir | \
! RUN:   FileCheck %t/ranges.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -module-dir %t %t/mapper.f90 -o /dev/null
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -J %t %t/consumer.f90 -o - | FileCheck %t/consumer.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   %t/cleanup.f90 -o - | FileCheck %t/cleanup.f90
! RUN: %flang_fc1 -emit-fir -O2 -fopenmp -fopenmp-version=52 \
! RUN:   %t/cleanup.f90 -o - | FileCheck %t/cleanup.f90 --check-prefix=FIR

!--- maps.f90
subroutine declare_mapper_mapper_references()
  type :: t
    integer :: n
    integer :: idx(4)
    integer :: a(4)
  end type
  !$omp declare mapper(m: t :: v) map(iterator(i = 1:v%n): v%a(v%idx(i)))
end

subroutine declare_mapper_scalar_parent()
  type :: inner_t
    integer :: a(2)
  end type
  type :: outer_t
    type(inner_t) :: p(3)
  end type
  !$omp declare mapper(m: outer_t :: v) map(iterator(i = 1:2): v%p(2)%a(i))
end

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
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] uniq_name("_QFdeclare_mapper_iteratorEv")
! CHECK:   %[[IT:.*]] = omp.iterator(%[[IV:.*]]: i32) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:     fir.store %[[IV]] to %[[IV_MEM:.*]] : !fir.ref<i32>
! CHECK:     %[[IV_DECL:.*]]:2 = hlfir.declare %[[IV_MEM]]
! CHECK:     %[[A:.*]] = hlfir.designate %[[DECL]]#0{"a"} {{.*}} : (!fir.ref<!fir.type<_QFdeclare_mapper_iteratorTs{{.*}}>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:     %[[IV_LD:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
! CHECK:     %[[IV_I64:.*]] = fir.convert %[[IV_LD]] : (i32) -> i64
! CHECK:     %[[IV_IDX:.*]] = fir.convert %[[IV_I64]] : (i64) -> index
! CHECK:     %[[LB:.*]] = arith.subi %[[IV_IDX]], %{{.*}} : index
! CHECK:     %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[LB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[A]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) name("") -> !llvm.ptr
! CHECK:     omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK:   } inclusive -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_allocatablem
! CHECK: ^bb0(%[[ARG2:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_allocatableTt{{.*}}>):
! CHECK:   %[[DECL2:.*]]:2 = hlfir.declare %[[ARG2]] uniq_name("_QFdeclare_mapper_allocatableEv")
! CHECK:   %[[IT2:.*]] = omp.iterator(%[[IV2:.*]]: i32) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:     %[[BOX_REF:.*]] = hlfir.designate %[[DECL2]]#0{"a"}{{.*}} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:     %[[BOX:.*]] = fir.load %[[BOX_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:     %[[DIMS0:.*]]:3 = fir.box_dims %[[BOX]], %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:     %[[DIMS1:.*]]:3 = fir.box_dims %[[BOX]], %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:     %[[BOUNDS2:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%[[DIMS1]]#1 : index) stride(%[[DIMS1]]#2 : index) start_idx(%[[DIMS0]]#0 : index) stride_in_bytes(true)
! CHECK:     %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:     %[[MAP2:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.heap<!fir.array<?xi32>>, i32) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS2]]) name("") -> !llvm.ptr
! CHECK:     omp.yield(%[[MAP2]] : !llvm.ptr)
! CHECK:   } inclusive -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT2]] : !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_pointerm
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_pointerTt{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] uniq_name("_QFdeclare_mapper_pointerEv")
! CHECK:   %[[IT:.*]] = omp.iterator(%{{.*}}: i32) = ({{.*}}) {
! CHECK:     %[[BOX_REF:.*]] = hlfir.designate %[[DECL]]#0{"a"}{{.*}} -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:     %[[BOX:.*]] = fir.load %[[BOX_REF]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:     %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.ptr<!fir.array<?xi32>>, i32) map_clauses(tofrom) capture(ByRef) bounds(%{{.*}}) name("") -> !llvm.ptr
! CHECK:   } inclusive -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_2dm
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_2dTt{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] uniq_name("_QFdeclare_mapper_2dEv")
! CHECK:   %[[I_LB:.*]] = arith.constant 1 : i32
! CHECK:   %[[I_UB:.*]] = arith.constant 4 : i32
! CHECK:   %[[I_STEP:.*]] = arith.constant 1 : i32
! CHECK:   %[[J_LB:.*]] = arith.constant 1 : i32
! CHECK:   %[[J_UB:.*]] = arith.constant 6 : i32
! CHECK:   %[[J_STEP:.*]] = arith.constant 1 : i32
! CHECK:   %[[IT:.*]] = omp.iterator(%[[IV_I:.*]]: i32,
! CHECK-SAME: %[[IV_J:.*]]: i32) =
! CHECK-SAME: (%[[I_LB]] to %[[I_UB]] step %[[I_STEP]],
! CHECK-SAME: %[[J_LB]] to %[[J_UB]] step %[[J_STEP]]) {
! CHECK:     fir.store %[[IV_I]] to %[[IV_I_MEM:.*]] : !fir.ref<i32>
! CHECK:     %[[IV_I_DECL:.*]]:2 = hlfir.declare %[[IV_I_MEM]]
! CHECK:     fir.store %[[IV_J]] to %[[IV_J_MEM:.*]] : !fir.ref<i32>
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
! CHECK:   } inclusive -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_sectionm
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_sectionTt{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] uniq_name("_QFdeclare_mapper_sectionEv")
! CHECK:   %[[IT:.*]] = omp.iterator(%{{.*}}: i32) = ({{.*}}) {
! CHECK:     %[[A:.*]] = hlfir.designate %[[DECL]]#0{"a"}{{.*}} -> !fir.ref<!fir.array<10xi32>>
! CHECK:     %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
! CHECK:     %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[A]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) name("") -> !llvm.ptr
! CHECK:   } inclusive -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_multim
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_multiTt{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] uniq_name("_QFdeclare_mapper_multiEv")
! CHECK:   %[[IT_A:.*]] = omp.iterator(%{{.*}}: i32) = ({{.*}}) {
! CHECK:     %[[A:.*]] = hlfir.designate %[[DECL]]#0{"a"}{{.*}} -> !fir.ref<!fir.array<10xi32>>
! CHECK:     %[[MAP_A:.*]] = omp.map.info var_ptr(%[[A]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%{{.*}}) name("") -> !llvm.ptr
! CHECK:   } inclusive -> !omp.iterated<!llvm.ptr>
! CHECK:   %[[IT_B:.*]] = omp.iterator(%{{.*}}: i32) = ({{.*}}) {
! CHECK:     %[[B:.*]] = hlfir.designate %[[DECL]]#0{"b"}{{.*}} -> !fir.ref<!fir.array<10xi32>>
! CHECK:     %[[MAP_B:.*]] = omp.map.info var_ptr(%[[B]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%{{.*}}) name("") -> !llvm.ptr
! CHECK:   } inclusive -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT_A]], %[[IT_B]] : !omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_alloc_sectionm
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_alloc_sectionTt{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] uniq_name("_QFdeclare_mapper_alloc_sectionEv")
! CHECK:   %[[IT:.*]] = omp.iterator(%{{.*}}: i32) = ({{.*}}) {
! CHECK:     %[[BOX_REF:.*]] = hlfir.designate %[[DECL]]#0{"a"}{{.*}} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:     %[[BOX:.*]] = fir.load %[[BOX_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:     %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
! CHECK:     %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index) stride_in_bytes(true)
! CHECK:     %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.heap<!fir.array<?xi32>>, i32) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) name("") -> !llvm.ptr
! CHECK:   } inclusive -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_nondefault_lbm
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_nondefault_lbTt{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] uniq_name("_QFdeclare_mapper_nondefault_lbEv")
! CHECK:   %[[LB:.*]] = arith.constant -2 : i32
! CHECK:   %[[UB:.*]] = arith.constant 6 : i32
! CHECK:   %[[STEP:.*]] = arith.constant 1 : i32
! CHECK:   %[[IT:.*]] = omp.iterator(%[[IV:.*]]: i32) =
! CHECK-SAME: (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:     fir.store %[[IV]] to %[[IV_MEM:.*]] : !fir.ref<i32>
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
! CHECK-SAME: stride_in_bytes(true)
! CHECK:     %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[BASE]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) name("") -> !llvm.ptr
! CHECK:   } inclusive -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! A scalar parent element keeps a single ranked part in the array base.
! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_scalar_parentm
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdeclare_mapper_scalar_parentEv")
! CHECK:   %[[IT:.*]] = omp.iterator(%{{.*}}: i32) =
! CHECK:     fir.store %{{.*}} to %[[IV_MEM:.*]] : !fir.ref<i32>
! CHECK:     %[[IV_DECL:.*]]:2 = hlfir.declare %[[IV_MEM]]
! CHECK:     %[[P_IDX:.*]] = arith.constant 2 : index
! CHECK:     %[[P:.*]] = hlfir.designate %[[DECL]]#0{"p"} <%{{.*}}> (%[[P_IDX]])
! CHECK:     %[[A:.*]] = hlfir.designate %[[P]]{"a"} {{.*}} -> !fir.ref<!fir.array<2xi32>>
! CHECK:     %[[IV:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
! CHECK:     %[[IV_I64:.*]] = fir.convert %[[IV]] : (i32) -> i64
! CHECK:     %[[IV_IDX:.*]] = fir.convert %[[IV_I64]] : (i64) -> index
! CHECK:     %[[LB:.*]] = arith.subi %[[IV_IDX]], %{{.*}} : index
! CHECK:     %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[LB]] : index)
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[A]] : !fir.ref<!fir.array<2xi32>>, !fir.array<2xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]])
! CHECK:     omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! Ranges and subscripts may reference the mapper variable.
! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_mapper_referencesm
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdeclare_mapper_mapper_referencesEv")
! CHECK:   %[[N_REF:.*]] = hlfir.designate %[[DECL]]#0{"n"}
! CHECK:   %[[N:.*]] = fir.load %[[N_REF]] : !fir.ref<i32>
! CHECK:   %[[IT:.*]] = omp.iterator(%{{.*}}: i32) = (%{{.*}} to %[[N]] step %{{.*}}) {
! CHECK:     %[[A:.*]] = hlfir.designate %[[DECL]]#0{"a"}
! CHECK:     %[[IDX:.*]] = hlfir.designate %[[DECL]]#0{"idx"}
! CHECK:     %[[SUB:.*]] = fir.load %[[IDX]] : !fir.ref<i32>
! CHECK:     %[[SUB_I64:.*]] = fir.convert %[[SUB]] : (i32) -> i64
! CHECK:     %[[SUB_IDX:.*]] = fir.convert %[[SUB_I64]] : (i64) -> index
! CHECK:     %[[LB:.*]] = arith.subi %[[SUB_IDX]], %{{.*}} : index
! CHECK:     %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[LB]] : index)
! CHECK:     omp.map.info var_ptr(%[[A]] : {{.*}}) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]])
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

!--- ranges.f90
subroutine mapper_ranges()
  type :: t
    integer :: a(4), b(4)
  end type
  !$omp declare mapper(m: t :: v) &
  !$omp& map(iterator(i=1:4, j=2:1): v%a(i), v%b(j))
end

! CHECK-LABEL: omp.declare_mapper @_QQFmapper_rangesm
! CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : i32
! CHECK-DAG: %[[FOUR:.*]] = arith.constant 4 : i32
! CHECK: %[[A:.*]] = omp.iterator(%{{[^:]+}}: i32) = (%{{[^ ]+}} to %[[FOUR]] step
! CHECK: hlfir.designate %{{.*}}{"a"}
! CHECK: omp.map.info
! CHECK: } inclusive -> !omp.iterated<!llvm.ptr>
! CHECK: %[[B:.*]] = omp.iterator(%{{[^:]+}}: i32) = (%[[TWO]] to %{{[^ ]+}} step
! CHECK: hlfir.designate %{{.*}}{"b"}
! CHECK: omp.map.info
! CHECK: } inclusive -> !omp.iterated<!llvm.ptr>
! CHECK: omp.declare_mapper.info map_iterated(%[[A]], %[[B]]

!--- mapper.f90
module iterator_mapper_mod
  type :: t
    integer :: a(10)
  end type
  !$omp declare mapper(m: t :: v) map(iterator(i = 1:10): v%a(i))
end module

!--- consumer.f90
subroutine use_iterator_mapper_mod
  use iterator_mapper_mod
  integer :: k
  k = 1
end subroutine

! CHECK-LABEL: omp.declare_mapper @_QQMiterator_mapper_modm
! CHECK: %[[ITER:.*]] = omp.iterator
! CHECK: %[[MAP:.*]] = omp.map.info
! CHECK: omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: omp.declare_mapper.info map_iterated(%[[ITER]]

!--- cleanup.f90
! Parentheses add no control flow to the single-block mapper region.
subroutine declare_mapper_paren_range()
  type :: t
    integer :: n
    integer :: a(4)
  end type
  !$omp declare mapper(m: t :: v) map(iterator(i = 1:(v%n)): v%a(i))
end

! Cleanups for the range expression stay inside the single-block mapper region.
subroutine declare_mapper_range_cleanup()
  interface
    integer function extent_of(n)
      integer :: n
    end function
  end interface
  type :: t
    integer :: n
    integer :: a(4)
  end type
  !$omp declare mapper(m: t :: v) map(iterator(i = 1:extent_of(v%n + 1)): v%a(i))
end

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_range_cleanupm
! CHECK:   %[[TMP:.*]]:3 = hlfir.associate
! CHECK:   fir.call @_QPextent_of(%[[TMP]]#0)
! CHECK:   %[[IT:.*]] = omp.iterator
! CHECK:   } inclusive -> !omp.iterated<!llvm.ptr>
! CHECK-NEXT: hlfir.end_associate %[[TMP]]#1, %[[TMP]]#2
! CHECK-NEXT: omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! CHECK-LABEL: omp.declare_mapper @_QQFdeclare_mapper_paren_rangem
! CHECK:   %[[N:.*]] = hlfir.no_reassoc
! CHECK:   %[[IT:.*]] = omp.iterator(%{{[^:]+}}: i32) = (%{{[^ ]+}} to %[[N]] step
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! The -O2 HLFIR-to-FIR pipeline adds no control flow to the mapper regions.
! FIR-LABEL: omp.declare_mapper @_QQFdeclare_mapper_range_cleanupm
! FIR-NOT:     {{fir\.(if|do_loop|iterate_while)}}
! FIR:         fir.call @_QPextent_of
! FIR-NOT:     {{fir\.(if|do_loop|iterate_while)}}
! FIR:         omp.declare_mapper.info map_iterated

! FIR-LABEL: omp.declare_mapper @_QQFdeclare_mapper_paren_rangem
! FIR-NOT:     {{fir\.(if|do_loop|iterate_while)}}
! FIR:         %[[N:.*]] = fir.no_reassoc
! FIR:         omp.iterator(%{{[^:]+}}: i32) = (%{{[^ ]+}} to %[[N]] step
! FIR-NOT:     {{fir\.(if|do_loop|iterate_while)}}
! FIR:         omp.declare_mapper.info map_iterated
