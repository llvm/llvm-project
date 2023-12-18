// RUN: fir-opt --bufferize-hlfir %s | FileCheck %s

func.func @test_poly_expr_without_associate() {
  %5 = fir.alloca !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>> {bindc_name = "r", uniq_name = "_QFtestEr"}
  %8:2 = hlfir.declare %5 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtestEr"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>)
  %26 = fir.undefined !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>
  %27:2 = hlfir.declare %26 {uniq_name = ".tmp.intrinsic_result"} : (!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>) -> (!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>, !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>)
  %28 = hlfir.as_expr %27#0 : (!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>) -> !hlfir.expr<!fir.type<_QFtestTt{c:i32}>?>
  hlfir.assign %28 to %8#0 realloc : !hlfir.expr<!fir.type<_QFtestTt{c:i32}>?>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>
  hlfir.destroy %28 : !hlfir.expr<!fir.type<_QFtestTt{c:i32}>?>
  return
}
// CHECK-LABEL:   func.func @test_poly_expr_without_associate() {
// CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>> {bindc_name = "r", uniq_name = "_QFtestEr"}
// CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtestEr"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>)
// CHECK:           %[[VAL_3:.*]] = fir.undefined !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>) -> (!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>, !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>)
// CHECK:           %[[VAL_5:.*]] = fir.zero_bits !fir.heap<!fir.type<_QFtestTt{c:i32}>>
// CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_5]] : (!fir.heap<!fir.type<_QFtestTt{c:i32}>>) -> !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>
// CHECK:           fir.store %[[VAL_6]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>
// CHECK:           %[[VAL_7:.*]] = arith.constant true
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = ".tmp"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>)
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_4]]#1 : (!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>) -> !fir.box<none>
// CHECK:           %[[VAL_12:.*]] = fir.call @_FortranAAllocatableApplyMold(%[[VAL_10]], %[[VAL_11]], %[[VAL_9]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> none
// CHECK:           hlfir.assign %[[VAL_4]]#0 to %[[VAL_8]]#0 realloc temporary_lhs : !fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>
// CHECK:           %[[VAL_13:.*]] = fir.undefined tuple<!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>, i1>
// CHECK:           %[[VAL_14:.*]] = fir.insert_value %[[VAL_13]], %[[VAL_7]], [1 : index] : (tuple<!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>, i1>, i1) -> tuple<!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>, i1>
// CHECK:           %[[VAL_15:.*]] = fir.insert_value %[[VAL_14]], %[[VAL_8]]#0, [0 : index] : (tuple<!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>, i1>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>) -> tuple<!fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>, i1>
// CHECK:           hlfir.assign %[[VAL_8]]#0 to %[[VAL_2]]#0 realloc : !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>
// CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>>
// CHECK:           %[[VAL_17:.*]] = fir.box_addr %[[VAL_16]] : (!fir.class<!fir.heap<!fir.type<_QFtestTt{c:i32}>>>) -> !fir.heap<!fir.type<_QFtestTt{c:i32}>>
// CHECK:           fir.freemem %[[VAL_17]] : !fir.heap<!fir.type<_QFtestTt{c:i32}>>
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
  %9:3 = hlfir.associate %6(%8) {uniq_name = ".tmp.assign"} : (!hlfir.expr<?x!fir.type<_QMtest_typeTt1{i:i32}>?>, !fir.shape<1>) -> (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>, i1)
  %10 = fir.convert %0 : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>) -> !fir.box<none>
  %11 = fir.call @_FortranADestroy(%10) fastmath<contract> : (!fir.box<none>) -> none
  %c3 = arith.constant 3 : index
  %12 = fir.shape %c3 : (index) -> !fir.shape<1>
  %c1 = arith.constant 1 : index
  fir.do_loop %arg2 = %c1 to %c3 step %c1 {
    %13 = hlfir.designate %2#0 (%arg2)  : (!fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>>, index) -> !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>
    %14 = hlfir.designate %9#0 (%arg2)  : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>, index) -> !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>
    fir.dispatch "assign"(%13 : !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>) (%13, %14 : !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>, !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>) {pass_arg_pos = 0 : i32}
  }
  hlfir.end_associate %9#1, %9#2 : !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>, i1
  return
}
// CHECK-LABEL:   func.func @test_poly_expr_with_associate(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>> {fir.bindc_name = "v2"}) {
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>> {bindc_name = ".result"}
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtestEv2"} : (!fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>>) -> (!fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>>, !fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>>)
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>)
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_6]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_9:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_9]](%[[VAL_11]]) : (!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>, !fir.shape<1>) -> !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>
// CHECK:           fir.store %[[VAL_12]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>
// CHECK:           %[[VAL_13:.*]] = arith.constant true
// CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = ".tmp"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>)
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_5]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>) -> !fir.box<none>
// CHECK:           %[[VAL_18:.*]] = fir.call @_FortranAAllocatableApplyMold(%[[VAL_16]], %[[VAL_17]], %[[VAL_15]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> none
// CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_14]]#0 realloc temporary_lhs : !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>
// CHECK:           %[[VAL_19:.*]] = fir.undefined tuple<!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>, i1>
// CHECK:           %[[VAL_20:.*]] = fir.insert_value %[[VAL_19]], %[[VAL_13]], [1 : index] : (tuple<!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>, i1>, i1) -> tuple<!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>, i1>
// CHECK:           %[[VAL_21:.*]] = fir.insert_value %[[VAL_20]], %[[VAL_14]]#0, [0 : index] : (tuple<!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>, i1>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>) -> tuple<!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>, i1>
// CHECK:           %[[VAL_22:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_23:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_22]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_24:.*]] = fir.shape %[[VAL_23]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_25:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>
// CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_14]]#1 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>
// CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>>) -> !fir.box<none>
// CHECK:           %[[VAL_28:.*]] = fir.call @_FortranADestroy(%[[VAL_27]]) fastmath<contract> : (!fir.box<none>) -> none
// CHECK:           %[[VAL_29:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_30:.*]] = fir.shape %[[VAL_29]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_31:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_32:.*]] = %[[VAL_31]] to %[[VAL_29]] step %[[VAL_31]] {
// CHECK:             %[[VAL_33:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_32]])  : (!fir.class<!fir.array<3x!fir.type<_QMtest_typeTt1{i:i32}>>>, index) -> !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>
// CHECK:             %[[VAL_34:.*]] = hlfir.designate %[[VAL_25]] (%[[VAL_32]])  : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>, index) -> !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>
// CHECK:             fir.dispatch "assign"(%[[VAL_33]] : !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>) (%[[VAL_33]], %[[VAL_34]] : !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>, !fir.class<!fir.type<_QMtest_typeTt1{i:i32}>>) {pass_arg_pos = 0 : i32}
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = fir.box_addr %[[VAL_26]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>>) -> !fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>
// CHECK:           fir.freemem %[[VAL_35]] : !fir.heap<!fir.array<?x!fir.type<_QMtest_typeTt1{i:i32}>>>
// CHECK:           return
// CHECK:         }
