// RUN: fir-opt --bufferize-hlfir %s | FileCheck %s

func.func @test_poly_expr_without_associate() {
  %5 = fir.alloca !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>> {bindc_name = "r", uniq_name = "_QFtestEr"}
  %8:2 = hlfir.declare %5 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtestEr"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>)
  %26 = fir.undefined !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>
  %27:2 = hlfir.declare %26 {uniq_name = ".tmp.intrinsic_result"} : (!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>) -> (!fir.class<!fir.type<_QFtestTt{c:i32}>>, !fir.class<!fir.type<_QFtestTt{c:i32}>>)
  %28 = hlfir.as_expr %27#0 : (!fir.class<!fir.type<_QFtestTt{c:i32}>>) -> !hlfir.expr<!fir.type<_QFtestTt{c:i32}>?>
  hlfir.assign %28 to %8#0 realloc : !hlfir.expr<!fir.type<_QFtestTt{c:i32}>?>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>
  hlfir.destroy %28 : !hlfir.expr<!fir.type<_QFtestTt{c:i32}>?>
  return
}
// CHECK-LABEL:   func.func @test_poly_expr_without_associate() {
// CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>> {bindc_name = "r", uniq_name = "_QFtestEr"}
// CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtestEr"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>)
// CHECK:           %[[VAL_3:.*]] = fir.undefined !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>) -> (!fir.class<!fir.type<_QFtestTt{c:i32}>>, !fir.class<!fir.type<_QFtestTt{c:i32}>>)
// CHECK:           %[[VAL_5:.*]] = fir.zero_bits !fir.heap<!fir.type<_QFtestTt{c:i32}>>
// CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_5]] source_box %[[VAL_4]]#0 : (!fir.heap<!fir.type<_QFtestTt{c:i32}>>, !fir.class<!fir.type<_QFtestTt{c:i32}>>) -> !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>
// CHECK:           fir.store %[[VAL_6]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>
// CHECK:           %[[VAL_7:.*]] = fir.zero_bits !fir.ref<none>
// CHECK:           %[[VAL_11:.*]] = arith.constant false
// CHECK:           %[[VAL_12:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<none>) -> !fir.ref<i64>
// CHECK:           %[[VAL_16:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_13]], %[[VAL_14]], %[[VAL_11]], %[[VAL_12:.*]], {{.*}}
// CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>
// CHECK:           %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_17]] {uniq_name = ".tmp"} : (!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>) -> (!fir.class<!fir.type<_QFtestTt{c:i32}>>, !fir.class<!fir.type<_QFtestTt{c:i32}>>)
// CHECK:           hlfir.assign %[[VAL_4]]#0 to %[[VAL_18]]#0 temporary_lhs : !fir.class<!fir.type<_QFtestTt{c:i32}>>, !fir.class<!fir.type<_QFtestTt{c:i32}>>
// CHECK:           %[[VAL_19:.*]] = arith.constant true
// CHECK:           %[[VAL_20:.*]] = fir.undefined tuple<!fir.class<!fir.type<_QFtestTt{c:i32}>>, i1>
// CHECK:           %[[VAL_21:.*]] = fir.insert_value %[[VAL_20]], %[[VAL_19]], [1 : index] : (tuple<!fir.class<!fir.type<_QFtestTt{c:i32}>>, i1>, i1) -> tuple<!fir.class<!fir.type<_QFtestTt{c:i32}>>, i1>
// CHECK:           %[[VAL_22:.*]] = fir.insert_value %[[VAL_21]], %[[VAL_18]]#0, [0 : index] : (tuple<!fir.class<!fir.type<_QFtestTt{c:i32}>>, i1>, !fir.class<!fir.type<_QFtestTt{c:i32}>>) -> tuple<!fir.class<!fir.type<_QFtestTt{c:i32}>>, i1>
// CHECK:           hlfir.assign %[[VAL_18]]#0 to %[[VAL_2]]#0 realloc : !fir.class<!fir.type<_QFtestTt{c:i32}>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>
// CHECK:           %[[VAL_23:.*]] = fir.box_addr %[[VAL_18]]#0 : (!fir.class<!fir.type<_QFtestTt{c:i32}>>) -> !fir.heap<!fir.type<_QFtestTt{c:i32}>>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_18]]#0 : (!fir.class<!fir.type<_QFtestTt{c:i32}>>) -> !fir.box<none>
// CHECK:           fir.call @_FortranADestroyWithoutFinalization(%[[VAL_24]]) : (!fir.box<none>) -> ()
// CHECK:           fir.freemem %[[VAL_23]] : !fir.heap<!fir.type<_QFtestTt{c:i32}>>
// CHECK:           return
// CHECK:         }

func.func @test_poly_expr_with_associate(%arg1: !fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>> {fir.bindc_name = "v2"}) {
  %0 = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>> {bindc_name = ".result"}
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFtestEv2"} : (!fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>>) -> (!fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>>, !fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>>)
  %4:2 = hlfir.declare %0 {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>)
  %5 = fir.load %4#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>
  %6 = hlfir.as_expr %5 : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>) -> !hlfir.expr<?x!fir.type<_QMtest_typeTt1{i:i32}>?>
  %c0 = arith.constant 0 : index
  %7:3 = fir.box_dims %5, %c0 : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>, index) -> (index, index, index)
  %8 = fir.shape %7#1 : (index) -> !fir.shape<1>
  %9:3 = hlfir.associate %6(%8) {uniq_name = ".tmp.assign"} : (!hlfir.expr<?x!fir.type<_QMtest_typeTt1{i:i32}>?>, !fir.shape<1>) -> (!fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>, !fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>, i1)
  %10 = fir.convert %0 : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>) -> !fir.box<none>
  fir.call @_FortranADestroy(%10) fastmath<contract> : (!fir.box<none>) -> ()
  %c3 = arith.constant 3 : index
  %12 = fir.shape %c3 : (index) -> !fir.shape<1>
  %c1 = arith.constant 1 : index
  fir.do_loop %arg2 = %c1 to %c3 step %c1 {
    %13 = hlfir.designate %2#0 (%arg2)  : (!fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>>, index) -> !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>
    %14 = hlfir.designate %9#0 (%arg2)  : (!fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>, index) -> !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>
    fir.dispatch "assign"(%13 : !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>) (%13, %14 : !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>, !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>) {pass_arg_pos = 0 : i32}
  }
  hlfir.end_associate %9#1, %9#2 : !fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>, i1
  return
}
// CHECK-LABEL:   func.func @test_poly_expr_with_associate(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>> {fir.bindc_name = "v2"}) {
// CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>> {bindc_name = ".result"}
// CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFtestEv2"} : (!fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>>) -> (!fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>>, !fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>>)
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>)
// CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_4]], %[[VAL_5]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_8:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>
// CHECK:           %[[VAL_9:.*]] = fir.embox %[[VAL_8]](%[[VAL_7]]) source_box %[[VAL_4]] : (!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>, !fir.shape<1>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>) -> !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>
// CHECK:           fir.store %[[VAL_9]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>
// CHECK:           %[[VAL_10:.*]] = fir.zero_bits !fir.ref<none>
// CHECK:           %[[VAL_14:.*]] = arith.constant false
// CHECK:           %[[VAL_15:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<none>) -> !fir.ref<i64>
// CHECK:           %[[VAL_19:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_16]], %[[VAL_17]], %[[VAL_14]], %[[VAL_15]], {{.*}}
// CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>
// CHECK:           %[[VAL_21:.*]]:2 = hlfir.declare %[[VAL_20]] {uniq_name = ".tmp"} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>) -> (!fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>, !fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>)
// CHECK:           hlfir.assign %[[VAL_4]] to %[[VAL_21]]#0 temporary_lhs : !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>, !fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>
// CHECK:           %[[VAL_22:.*]] = arith.constant true
// CHECK:           %[[VAL_23:.*]] = fir.undefined tuple<!fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>, i1>
// CHECK:           %[[VAL_24:.*]] = fir.insert_value %[[VAL_23]], %[[VAL_22]], [1 : index] : (tuple<!fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>, i1>, i1) -> tuple<!fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>, i1>
// CHECK:           %[[VAL_25:.*]] = fir.insert_value %[[VAL_24]], %[[VAL_21]]#0, [0 : index] : (tuple<!fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>, i1>, !fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>) -> tuple<!fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>, i1>
// CHECK:           %[[VAL_26:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_27:.*]]:3 = fir.box_dims %[[VAL_4]], %[[VAL_26]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_28:.*]] = fir.shape %[[VAL_27]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>) -> !fir.box<none>
// CHECK:           fir.call @_FortranADestroy(%[[VAL_29]]) fastmath<contract> : (!fir.box<none>) -> ()
// CHECK:           %[[VAL_30:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_31:.*]] = fir.shape %[[VAL_30]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_32:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_33:.*]] = %[[VAL_32]] to %[[VAL_30]] step %[[VAL_32]] {
// CHECK:             %[[VAL_34:.*]] = hlfir.designate %[[VAL_2]]#0 (%[[VAL_33]])  : (!fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>>, index) -> !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>
// CHECK:             %[[VAL_35:.*]] = hlfir.designate %[[VAL_21]]#0 (%[[VAL_33]])  : (!fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>, index) -> !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>
// CHECK:             fir.dispatch "assign"(%[[VAL_34]] : !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>) (%[[VAL_34]], %[[VAL_35]] : !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>, !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>) {pass_arg_pos = 0 : i32}
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = fir.box_addr %[[VAL_21]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>) -> !fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>
// CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_21]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>) -> !fir.box<none>
// CHECK:           fir.call @_FortranADestroyWithoutFinalization(%[[VAL_37]]) : (!fir.box<none>) -> ()
// CHECK:           fir.freemem %[[VAL_36]] : !fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>
// CHECK:           return
// CHECK:         }
