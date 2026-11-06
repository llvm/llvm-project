! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.

subroutine query_array_element()
  use iso_fortran_env, only: event_type
  type(event_type), save :: ev(3)[*]
  integer :: count
  call event_query(ev(2), count)
end subroutine

! COARRAY-LABEL: func.func @_QPquery_array_element() {
! COARRAY: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY: %[[VAL_1:.*]] = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_array_elementEcount"}>
! COARRAY: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFquery_array_elementEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_3:.*]] = fir.address_of(@_QFquery_array_elementEev) : !fir.ref<!fir.box<!fir.heap<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>, corank:1>>
! COARRAY: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFquery_array_elementEev") : {{.*}}
! COARRAY: %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>, corank:1>>
! COARRAY: %[[C2:.*]] = arith.constant 2 : index
! COARRAY: %[[VAL_6:.*]] = hlfir.designate %[[VAL_5]] (%[[C2]]) {{.*}} -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_7:.*]] = mif.event_query %[[VAL_6]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i32
! COARRAY: fir.store %[[VAL_7]] to %[[VAL_2]]#0 : !fir.ref<i32>
! COARRAY: return

subroutine query_array_in_loop()
  use iso_fortran_env, only: event_type
  type(event_type), save :: ev(4)[*]
  integer :: counts(4), stats(4), i
  do i = 1, 4
    call event_query(ev(i), counts(i), stats(i))
  end do
end subroutine

! COARRAY-LABEL: func.func @_QPquery_array_in_loop() {
! COARRAY: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY: %[[C4:.*]] = arith.constant 4 : index
! COARRAY: %[[VAL_1:.*]] = fir.alloca !fir.array<4xi32> <{bindc_name = "counts", uniq_name = "_QFquery_array_in_loopEcounts"}>
! COARRAY: %[[VAL_2:.*]] = fir.shape %[[C4]] : (index) -> !fir.shape<1>
! COARRAY: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_2]]) uniq_name("_QFquery_array_in_loopEcounts") : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi32>>, !fir.ref<!fir.array<4xi32>>)
! COARRAY: %[[VAL_4:.*]] = fir.address_of(@_QFquery_array_in_loopEev) : !fir.ref<!fir.box<!fir.heap<!fir.array<4x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>, corank:1>>
! COARRAY: %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] uniq_name("_QFquery_array_in_loopEev") : {{.*}}
! COARRAY: %[[VAL_6:.*]] = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFquery_array_in_loopEi"}>
! COARRAY: %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] uniq_name("_QFquery_array_in_loopEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[C4_0:.*]] = arith.constant 4 : index
! COARRAY: %[[VAL_8:.*]] = fir.alloca !fir.array<4xi32> <{bindc_name = "stats", uniq_name = "_QFquery_array_in_loopEstats"}>
! COARRAY: %[[VAL_9:.*]] = fir.shape %[[C4_0]] : (index) -> !fir.shape<1>
! COARRAY: %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_9]]) uniq_name("_QFquery_array_in_loopEstats") : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi32>>, !fir.ref<!fir.array<4xi32>>)
! COARRAY: %[[C1_I32:.*]] = arith.constant 1 : i32
! COARRAY: %[[C4_I32:.*]] = arith.constant 4 : i32
! COARRAY: %[[C1_I32_1:.*]] = arith.constant 1 : i32
! COARRAY: fir.do_loop %[[ARG0:.*]] = %[[C1_I32]] to %[[C4_I32]] step %[[C1_I32_1]] : i32 {
! COARRAY:   fir.store %[[ARG0]] to %[[VAL_7]]#0 : !fir.ref<i32>
! COARRAY:   %[[VAL_22:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<4x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>, corank:1>>
! COARRAY:   %[[VAL_23:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
! COARRAY:   %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i32) -> i64
! COARRAY:   %[[VAL_25:.*]] = hlfir.designate %[[VAL_22]] (%[[VAL_24]]) {{.*}} -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY:   %[[VAL_26:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
! COARRAY:   %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> i64
! COARRAY:   %[[VAL_28:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_27]]) {{.*}} -> !fir.ref<i32>
! COARRAY:   %[[VAL_29:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
! COARRAY:   %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
! COARRAY:   %[[VAL_31:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_30]]) {{.*}} -> !fir.ref<i32>
! COARRAY:   %[[VAL_32:.*]] = mif.event_query %[[VAL_25]] stat %[[VAL_31]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, !fir.ref<i32>) -> i32
! COARRAY:   fir.store %[[VAL_32]] to %[[VAL_28]] : !fir.ref<i32>
! COARRAY: }
! COARRAY: %[[VAL_11:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
! COARRAY: %[[VAL_12:.*]] = fir.convert %[[C4_I32]] : (i32) -> index
! COARRAY: %[[VAL_13:.*]] = fir.convert %[[C1_I32_1]] : (i32) -> index
! COARRAY: %[[C0:.*]] = arith.constant 0 : index
! COARRAY: %[[VAL_14:.*]] = arith.subi %[[VAL_12]], %[[VAL_11]] : index
! COARRAY: %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_13]] : index
! COARRAY: %[[VAL_16:.*]] = arith.divsi %[[VAL_15]], %[[VAL_13]] : index
! COARRAY: %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_16]], %[[C0]] : index
! COARRAY: %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[C0]], %[[VAL_16]] : index
! COARRAY: %[[VAL_19:.*]] = arith.muli %[[VAL_18]], %[[VAL_13]] : index
! COARRAY: %[[VAL_20:.*]] = arith.addi %[[VAL_11]], %[[VAL_19]] : index
! COARRAY: %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (index) -> i32
! COARRAY: fir.store %[[VAL_21]] to %[[VAL_7]]#0 : !fir.ref<i32>
! COARRAY: return

subroutine query_allocatable()
  use iso_fortran_env, only: event_type
  type(event_type), allocatable :: ev[:]
  integer :: count, stat
  allocate(ev[*])
  call event_query(ev, count)
  call event_query(ev, count, stat)
  deallocate(ev)
end subroutine

! COARRAY-LABEL: func.func @_QPquery_allocatable() {
! COARRAY: %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
! COARRAY: %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
! COARRAY: %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY: %[[VAL_3:.*]] = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_allocatableEcount"}>
! COARRAY: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFquery_allocatableEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_5:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1> {{.*}}uniq_name = "_QFquery_allocatableEev"{{.*}}
! COARRAY: %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_7:.*]] = fir.embox %[[VAL_6]] : ({{.*}}) -> !fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>
! COARRAY: fir.store %[[VAL_7]] to %[[VAL_5]] : {{.*}}
! COARRAY: %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_5]] uniq_name("_QFquery_allocatableEev") fortran_attrs<allocatable> : {{.*}}
! COARRAY: %[[VAL_9:.*]] = fir.alloca i32 <{bindc_name = "stat", uniq_name = "_QFquery_allocatableEstat"}>
! COARRAY: %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] uniq_name("_QFquery_allocatableEstat") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_11:.*]] = fir.absent !fir.box<none>
! COARRAY: %[[C1_I64:.*]] = arith.constant 1 : i64
! COARRAY: %[[C0:.*]] = arith.constant 0 : index
! COARRAY: %[[VAL_12:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! COARRAY: fir.store %[[C1_I64]] to %[[VAL_12]] : !fir.ref<i64>
! COARRAY: %[[VAL_13:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! COARRAY: %[[VAL_14:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! COARRAY: mif.alloc_coarray %[[VAL_8]]#0 lcobounds %[[VAL_13]] ucobounds %[[VAL_14]] errmsg %[[VAL_11]] uniq_name("_QFquery_allocatableEev") : {{.*}}
! COARRAY: %[[VAL_15:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_16:.*]] = fir.box_addr %[[VAL_15]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_17:.*]] = mif.event_query %[[VAL_16]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i32
! COARRAY: fir.store %[[VAL_17]] to %[[VAL_4]]#0 : !fir.ref<i32>
! COARRAY: %[[VAL_18:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_19:.*]] = fir.box_addr %[[VAL_18]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_20:.*]] = mif.event_query %[[VAL_19]] stat %[[VAL_10]]#0 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, !fir.ref<i32>) -> i32
! COARRAY: fir.store %[[VAL_20]] to %[[VAL_4]]#0 : !fir.ref<i32>
! COARRAY: %[[VAL_21:.*]] = fir.absent !fir.box<none>
! COARRAY: %[[VAL_22:.*]] = fir.absent !fir.ref<i32>
! COARRAY: mif.dealloc_coarray %[[VAL_8]]#0 stat %[[VAL_22]] errmsg %[[VAL_21]] : {{.*}}
! COARRAY: %[[VAL_23:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_24:.*]] = fir.box_addr %[[VAL_23]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
! COARRAY: %[[C0_I64:.*]] = arith.constant 0 : i64
! COARRAY: %[[VAL_26:.*]] = arith.cmpi ne, %[[VAL_25]], %[[C0_I64]] : i64
! COARRAY: fir.if %[[VAL_26]] {
! COARRAY:   %[[VAL_27:.*]] = fir.absent !fir.box<none>
! COARRAY:   %[[VAL_28:.*]] = fir.absent !fir.ref<i32>
! COARRAY:   mif.dealloc_coarray %[[VAL_8]]#0 stat %[[VAL_28]] errmsg %[[VAL_27]] : {{.*}}
! COARRAY: }
! COARRAY: return

subroutine query_allocatable_array(n)
  use iso_fortran_env, only: event_type
  integer, intent(in) :: n
  type(event_type), allocatable :: ev(:)[:]
  integer :: count, i
  allocate(ev(n)[*])
  do i = 1, n
    call event_query(ev(i), count)
  end do
  deallocate(ev)
end subroutine

! COARRAY-LABEL: func.func @_QPquery_allocatable_array(
! COARRAY-SAME: %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n", fir.fortran_attrs = #fir.var_attrs<intent_in>, fir.read_only}) {
! COARRAY: %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
! COARRAY: %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
! COARRAY: %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY: %[[VAL_3:.*]] = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_allocatable_arrayEcount"}>
! COARRAY: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFquery_allocatable_arrayEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_5:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>, corank:1> {{.*}}uniq_name = "_QFquery_allocatable_arrayEev"{{.*}}
! COARRAY: %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>
! COARRAY: %[[C0:.*]] = arith.constant 0 : index
! COARRAY: %[[VAL_7:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! COARRAY: %[[VAL_8:.*]] = fir.embox %[[VAL_6]](%[[VAL_7]]) : {{.*}} -> !fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>, corank:1>
! COARRAY: fir.store %[[VAL_8]] to %[[VAL_5]] : {{.*}}
! COARRAY: %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_5]] uniq_name("_QFquery_allocatable_arrayEev") fortran_attrs<allocatable> : {{.*}}
! COARRAY: %[[VAL_10:.*]] = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFquery_allocatable_arrayEi"}>
! COARRAY: %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] uniq_name("_QFquery_allocatable_arrayEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_12:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_2]] arg 1 uniq_name("_QFquery_allocatable_arrayEn") fortran_attrs<intent_in> : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_13:.*]] = fir.absent !fir.box<none>
! COARRAY: %[[C1:.*]] = arith.constant 1 : index
! COARRAY: %[[VAL_14:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<i32>
! COARRAY: %[[C0_I32:.*]] = arith.constant 0 : i32
! COARRAY: %[[VAL_15:.*]] = fir.convert %[[VAL_9]]#0 : {{.*}} -> !fir.ref<!fir.box<none>>
! COARRAY: %[[VAL_16:.*]] = fir.convert %[[C1]] : (index) -> i64
! COARRAY: %[[VAL_17:.*]] = fir.convert %[[VAL_14]] : (i32) -> i64
! COARRAY: fir.call @_FortranAAllocatableSetBounds(%[[VAL_15]], %[[C0_I32]], %[[VAL_16]], %[[VAL_17]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! COARRAY: %[[C1_I64:.*]] = arith.constant 1 : i64
! COARRAY: %[[C0_0:.*]] = arith.constant 0 : index
! COARRAY: %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0_0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! COARRAY: fir.store %[[C1_I64]] to %[[VAL_18]] : !fir.ref<i64>
! COARRAY: %[[VAL_19:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! COARRAY: %[[VAL_20:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! COARRAY: mif.alloc_coarray %[[VAL_9]]#0 lcobounds %[[VAL_19]] ucobounds %[[VAL_20]] errmsg %[[VAL_13]] uniq_name("_QFquery_allocatable_arrayEev") : {{.*}}
! COARRAY: %[[C1_I32:.*]] = arith.constant 1 : i32
! COARRAY: %[[VAL_21:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<i32>
! COARRAY: %[[C1_I32_1:.*]] = arith.constant 1 : i32
! COARRAY: fir.do_loop %[[ARG1:.*]] = %[[C1_I32]] to %[[VAL_21]] step %[[C1_I32_1]] : i32 {
! COARRAY:   fir.store %[[ARG1]] to %[[VAL_11]]#0 : !fir.ref<i32>
! COARRAY:   %[[VAL_39:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>, corank:1>>
! COARRAY:   %[[VAL_40:.*]] = fir.load %[[VAL_11]]#0 : !fir.ref<i32>
! COARRAY:   %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i32) -> i64
! COARRAY:   %[[VAL_42:.*]] = hlfir.designate %[[VAL_39]] (%[[VAL_41]]) {{.*}} -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY:   %[[VAL_43:.*]] = mif.event_query %[[VAL_42]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i32
! COARRAY:   fir.store %[[VAL_43]] to %[[VAL_4]]#0 : !fir.ref<i32>
! COARRAY: }
! COARRAY: %[[VAL_22:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
! COARRAY: %[[VAL_23:.*]] = fir.convert %[[VAL_21]] : (i32) -> index
! COARRAY: %[[VAL_24:.*]] = fir.convert %[[C1_I32_1]] : (i32) -> index
! COARRAY: %[[C0_2:.*]] = arith.constant 0 : index
! COARRAY: %[[VAL_25:.*]] = arith.subi %[[VAL_23]], %[[VAL_22]] : index
! COARRAY: %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_24]] : index
! COARRAY: %[[VAL_27:.*]] = arith.divsi %[[VAL_26]], %[[VAL_24]] : index
! COARRAY: %[[VAL_28:.*]] = arith.cmpi slt, %[[VAL_27]], %[[C0_2]] : index
! COARRAY: %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[C0_2]], %[[VAL_27]] : index
! COARRAY: %[[VAL_30:.*]] = arith.muli %[[VAL_29]], %[[VAL_24]] : index
! COARRAY: %[[VAL_31:.*]] = arith.addi %[[VAL_22]], %[[VAL_30]] : index
! COARRAY: %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (index) -> i32
! COARRAY: fir.store %[[VAL_32]] to %[[VAL_11]]#0 : !fir.ref<i32>
! COARRAY: %[[VAL_33:.*]] = fir.absent !fir.box<none>
! COARRAY: %[[VAL_34:.*]] = fir.absent !fir.ref<i32>
! COARRAY: mif.dealloc_coarray %[[VAL_9]]#0 stat %[[VAL_34]] errmsg %[[VAL_33]] : {{.*}}
! COARRAY: %[[VAL_35:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>, corank:1>>
! COARRAY: %[[VAL_36:.*]] = fir.box_addr %[[VAL_35]] : ({{.*}}) -> !fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>
! COARRAY: %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : ({{.*}}) -> i64
! COARRAY: %[[C0_I64:.*]] = arith.constant 0 : i64
! COARRAY: %[[VAL_38:.*]] = arith.cmpi ne, %[[VAL_37]], %[[C0_I64]] : i64
! COARRAY: fir.if %[[VAL_38]] {
! COARRAY:   %[[VAL_39:.*]] = fir.absent !fir.box<none>
! COARRAY:   %[[VAL_40:.*]] = fir.absent !fir.ref<i32>
! COARRAY:   mif.dealloc_coarray %[[VAL_9]]#0 stat %[[VAL_40]] errmsg %[[VAL_39]] : {{.*}}
! COARRAY: }
! COARRAY: return

subroutine query_class_allocatable()
  use iso_fortran_env, only: event_type
  class(event_type), allocatable :: ev[:]
  integer :: count
  allocate(event_type :: ev[*])
  call event_query(ev, count)
end subroutine

! COARRAY-LABEL: func.func @_QPquery_class_allocatable() {
! COARRAY: %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
! COARRAY: %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
! COARRAY: %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY: %[[VAL_3:.*]] = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_class_allocatableEcount"}>
! COARRAY: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFquery_class_allocatableEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_5:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1> {{.*}}uniq_name = "_QFquery_class_allocatableEev"{{.*}}
! COARRAY: %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_7:.*]] = fir.embox %[[VAL_6]] : ({{.*}}) -> !fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>
! COARRAY: fir.store %[[VAL_7]] to %[[VAL_5]] : {{.*}}
! COARRAY: %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_5]] uniq_name("_QFquery_class_allocatableEev") fortran_attrs<allocatable> : {{.*}}
! COARRAY: %[[VAL_9:.*]] = fir.absent !fir.box<none>
! COARRAY: %[[VAL_10:.*]] = fir.type_desc !fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>
! COARRAY: %[[C0_I32:.*]] = arith.constant 0 : i32
! COARRAY: %[[C0_I32_0:.*]] = arith.constant 0 : i32
! COARRAY: %[[VAL_11:.*]] = fir.convert %[[VAL_8]]#0 : {{.*}} -> !fir.ref<!fir.box<none>>
! COARRAY: %[[VAL_12:.*]] = fir.convert %[[VAL_10]] : {{.*}} -> !fir.ref<none>
! COARRAY: fir.call @_FortranAAllocatableInitDerivedForAllocate(%[[VAL_11]], %[[VAL_12]], %[[C0_I32]], %[[C0_I32_0]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> ()
! COARRAY: %[[C1_I64:.*]] = arith.constant 1 : i64
! COARRAY: %[[C0:.*]] = arith.constant 0 : index
! COARRAY: %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! COARRAY: fir.store %[[C1_I64]] to %[[VAL_13]] : !fir.ref<i64>
! COARRAY: %[[VAL_14:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! COARRAY: %[[VAL_15:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! COARRAY: mif.alloc_coarray %[[VAL_8]]#0 lcobounds %[[VAL_14]] ucobounds %[[VAL_15]] errmsg %[[VAL_9]] uniq_name("_QFquery_class_allocatableEev") : {{.*}}
! COARRAY: %[[VAL_16:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_17:.*]] = mif.event_query %[[VAL_16]] : (!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>) -> i32
! COARRAY: fir.store %[[VAL_17]] to %[[VAL_4]]#0 : !fir.ref<i32>
! COARRAY: %[[VAL_18:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_19:.*]] = fir.box_addr %[[VAL_18]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
! COARRAY: %[[C0_I64:.*]] = arith.constant 0 : i64
! COARRAY: %[[VAL_21:.*]] = arith.cmpi ne, %[[VAL_20]], %[[C0_I64]] : i64
! COARRAY: fir.if %[[VAL_21]] {
! COARRAY:   %[[VAL_22:.*]] = fir.absent !fir.box<none>
! COARRAY:   %[[VAL_23:.*]] = fir.absent !fir.ref<i32>
! COARRAY:   mif.dealloc_coarray %[[VAL_8]]#0 stat %[[VAL_23]] errmsg %[[VAL_22]] : {{.*}}
! COARRAY: }
! COARRAY: return
