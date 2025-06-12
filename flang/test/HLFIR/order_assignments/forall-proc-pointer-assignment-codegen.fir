// Test code generation of hlfir.region_assign representing procedure pointer
// assignments inside FORALL.

// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

!t=!fir.type<t{p:!fir.boxproc<() -> i32>}>
func.func @test_no_conflict(%arg0: !fir.ref<!fir.array<10x!t>> {fir.bindc_name = "x"}) {
  %c10_i64 = arith.constant 10 : i64
  %c1_i64 = arith.constant 1 : i64
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 {uniq_name = "x"} : (!fir.ref<!fir.array<10x!t>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10x!t>>, !fir.ref<!fir.array<10x!t>>)
  hlfir.forall lb {
    hlfir.yield %c1_i64 : i64
  } ub {
    hlfir.yield %c10_i64 : i64
  }  (%arg1: i64) {
    hlfir.region_assign {
      %3 = fir.address_of(@f1) : () -> i32
      %4 = fir.emboxproc %3 : (() -> i32) -> !fir.boxproc<() -> ()>
      hlfir.yield %4 : !fir.boxproc<() -> ()>
    } to {
      %3 = hlfir.designate %2#0 (%arg1)  : (!fir.ref<!fir.array<10x!t>>, i64) -> !fir.ref<!t>
      %4 = hlfir.designate %3{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!t>) -> !fir.ref<!fir.boxproc<() -> i32>>
      hlfir.yield %4 : !fir.ref<!fir.boxproc<() -> i32>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_no_conflict(
// CHECK:           %[[VAL_1:.*]] = arith.constant 10 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare{{.*}}"x"
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_2]] : (i64) -> index
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_1]] : (i64) -> index
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_10:.*]] = %[[VAL_7]] to %[[VAL_8]] step %[[VAL_9]] {
// CHECK:             %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (index) -> i64
// CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_11]])  : (!fir.ref<!fir.array<10x!fir.type<t{p:!fir.boxproc<() -> i32>}>>>, i64) -> !fir.ref<!fir.type<t{p:!fir.boxproc<() -> i32>}>>
// CHECK:             %[[VAL_13:.*]] = hlfir.designate %[[VAL_12]]{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<t{p:!fir.boxproc<() -> i32>}>>) -> !fir.ref<!fir.boxproc<() -> i32>>
// CHECK:             %[[VAL_14:.*]] = fir.address_of(@f1) : () -> i32
// CHECK:             %[[VAL_15:.*]] = fir.emboxproc %[[VAL_14]] : (() -> i32) -> !fir.boxproc<() -> ()>
// CHECK:             %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<() -> i32>
// CHECK:             fir.store %[[VAL_16]] to %[[VAL_13]] : !fir.ref<!fir.boxproc<() -> i32>>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @test_need_to_save_rhs(%arg0: !fir.ref<!fir.array<10x!t>> {fir.bindc_name = "x"}) {
  %c10_i64 = arith.constant 10 : i64
  %c1_i64 = arith.constant 1 : i64
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 {uniq_name = "x"} : (!fir.ref<!fir.array<10x!t>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10x!t>>, !fir.ref<!fir.array<10x!t>>)
  hlfir.forall lb {
    hlfir.yield %c1_i64 : i64
  } ub {
    hlfir.yield %c10_i64 : i64
  }  (%arg1: i64) {
    hlfir.region_assign {
      %3 = hlfir.designate %2#0 (%c10)  : (!fir.ref<!fir.array<10x!t>>, index) -> !fir.ref<!t>
      %4 = hlfir.designate %3{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!t>) -> !fir.ref<!fir.boxproc<() -> i32>>
      %5 = fir.load %4 : !fir.ref<!fir.boxproc<() -> i32>>
      hlfir.yield %5 : !fir.boxproc<() -> i32>
    } to {
      %3 = hlfir.designate %2#0 (%arg1)  : (!fir.ref<!fir.array<10x!t>>, i64) -> !fir.ref<!t>
      %4 = hlfir.designate %3{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!t>) -> !fir.ref<!fir.boxproc<() -> i32>>
      hlfir.yield %4 : !fir.ref<!fir.boxproc<() -> i32>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_need_to_save_rhs(
// CHECK:           %[[VAL_1:.*]] = fir.alloca i64
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<i64>>
// CHECK:           %[[VAL_3:.*]] = fir.alloca i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare{{.*}}x
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : i64
// CHECK:           fir.store %[[VAL_13]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK:           %[[VAL_19:.*]] = fir.call @_FortranACreateValueStack(
// CHECK:           fir.do_loop %[[VAL_20:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_12]] {
// CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (index) -> i64
// CHECK:             %[[VAL_22:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<10x!fir.type<t{p:!fir.boxproc<() -> i32>}>>>, index) -> !fir.ref<!fir.type<t{p:!fir.boxproc<() -> i32>}>>
// CHECK:             %[[VAL_23:.*]] = hlfir.designate %[[VAL_22]]{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<t{p:!fir.boxproc<() -> i32>}>>) -> !fir.ref<!fir.boxproc<() -> i32>>
// CHECK:             %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<!fir.boxproc<() -> i32>>
// CHECK:             %[[VAL_25:.*]] = fir.box_addr %[[VAL_24]] : (!fir.boxproc<() -> i32>) -> (() -> i32)
// CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (() -> i32) -> i64
// CHECK:             fir.store %[[VAL_26]] to %[[VAL_1]] : !fir.ref<i64>
// CHECK:             %[[VAL_27:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<i64>) -> !fir.box<i64>
// CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (!fir.box<i64>) -> !fir.box<none>
// CHECK:             fir.call @_FortranAPushValue(%[[VAL_19]], %[[VAL_28]]) : (!fir.llvm_ptr<i8>, !fir.box<none>) -> ()
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
// CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
// CHECK:           %[[VAL_31:.*]] = arith.constant 1 : index
// CHECK:           fir.store %[[VAL_13]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK:           fir.do_loop %[[VAL_32:.*]] = %[[VAL_29]] to %[[VAL_30]] step %[[VAL_31]] {
// CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
// CHECK:             %[[VAL_34:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_33]])  : (!fir.ref<!fir.array<10x!fir.type<t{p:!fir.boxproc<() -> i32>}>>>, i64) -> !fir.ref<!fir.type<t{p:!fir.boxproc<() -> i32>}>>
// CHECK:             %[[VAL_35:.*]] = hlfir.designate %[[VAL_34]]{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<t{p:!fir.boxproc<() -> i32>}>>) -> !fir.ref<!fir.boxproc<() -> i32>>
// CHECK:             %[[VAL_36:.*]] = fir.load %[[VAL_3]] : !fir.ref<i64>
// CHECK:             %[[VAL_37:.*]] = arith.addi %[[VAL_36]], %[[VAL_14]] : i64
// CHECK:             fir.store %[[VAL_37]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<i64>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             fir.call @_FortranAValueAt(%[[VAL_19]], %[[VAL_36]], %[[VAL_38]]) : (!fir.llvm_ptr<i8>, i64, !fir.ref<!fir.box<none>>) -> ()
// CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<i64>>>
// CHECK:             %[[VAL_40:.*]] = fir.box_addr %[[VAL_39]] : (!fir.box<!fir.heap<i64>>) -> !fir.heap<i64>
// CHECK:             %[[VAL_41:.*]] = fir.load %[[VAL_40]] : !fir.heap<i64>
// CHECK:             %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i64) -> (() -> i32)
// CHECK:             %[[VAL_43:.*]] = fir.emboxproc %[[VAL_42]] : (() -> i32) -> !fir.boxproc<() -> i32>
// CHECK:             fir.store %[[VAL_43]] to %[[VAL_35]] : !fir.ref<!fir.boxproc<() -> i32>>
// CHECK:           }
// CHECK:           fir.call @_FortranADestroyValueStack(%[[VAL_19]]) : (!fir.llvm_ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }

func.func @test_need_to_save_lhs(%arg0: !fir.ref<!fir.array<10x!t>>) {
  %c11_i64 = arith.constant 11 : i64
  %c10_i64 = arith.constant 10 : i64
  %c1_i64 = arith.constant 1 : i64
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 {uniq_name = "x"} : (!fir.ref<!fir.array<10x!t>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10x!t>>, !fir.ref<!fir.array<10x!t>>)
  hlfir.forall lb {
    hlfir.yield %c1_i64 : i64
  } ub {
    hlfir.yield %c10_i64 : i64
  }  (%arg1: i64) {
    hlfir.region_assign {
      %3 = fir.address_of(@f1) : () -> i32
      %4 = fir.emboxproc %3 : (() -> i32) -> !fir.boxproc<() -> ()>
      hlfir.yield %4 : !fir.boxproc<() -> ()>
    } to {
      %3 = arith.subi %c11_i64, %arg1 : i64
      %4 = hlfir.designate %2#0 (%3)  : (!fir.ref<!fir.array<10x!t>>, i64) -> !fir.ref<!t>
      %5 = hlfir.designate %4{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!t>) -> !fir.ref<!fir.boxproc<() -> i32>>
      %6 = fir.load %5 : !fir.ref<!fir.boxproc<() -> i32>>
      %7 = fir.box_addr %6 : (!fir.boxproc<() -> i32>) -> (() -> i32)
      %8 = fir.call %7() proc_attrs<pure> : () -> i32
      %9 = fir.convert %8 : (i32) -> i64
      %10 = hlfir.designate %2#0 (%9)  : (!fir.ref<!fir.array<10x!t>>, i64) -> !fir.ref<!t>
      %11 = hlfir.designate %10{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!t>) -> !fir.ref<!fir.boxproc<() -> i32>>
      hlfir.yield %11 : !fir.ref<!fir.boxproc<() -> i32>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_need_to_save_lhs(
// CHECK:           %[[VAL_1:.*]] = fir.alloca i64
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<i64>>
// CHECK:           %[[VAL_3:.*]] = fir.alloca i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 11 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 10 : i64
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_7:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare{{.*}}"x"
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
// CHECK:           %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_14:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i64
// CHECK:           fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK:           %[[VAL_20:.*]] = fir.call @_FortranACreateValueStack(
// CHECK:           fir.do_loop %[[VAL_21:.*]] = %[[VAL_11]] to %[[VAL_12]] step %[[VAL_13]] {
// CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (index) -> i64
// CHECK:             %[[VAL_23:.*]] = arith.subi %[[VAL_4]], %[[VAL_22]] : i64
// CHECK:             %[[VAL_24:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_23]])  : (!fir.ref<!fir.array<10x!fir.type<t{p:!fir.boxproc<() -> i32>}>>>, i64) -> !fir.ref<!fir.type<t{p:!fir.boxproc<() -> i32>}>>
// CHECK:             %[[VAL_25:.*]] = hlfir.designate %[[VAL_24]]{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<t{p:!fir.boxproc<() -> i32>}>>) -> !fir.ref<!fir.boxproc<() -> i32>>
// CHECK:             %[[VAL_26:.*]] = fir.load %[[VAL_25]] : !fir.ref<!fir.boxproc<() -> i32>>
// CHECK:             %[[VAL_27:.*]] = fir.box_addr %[[VAL_26]] : (!fir.boxproc<() -> i32>) -> (() -> i32)
// CHECK:             %[[VAL_28:.*]] = fir.call %[[VAL_27]]() proc_attrs<pure> : () -> i32
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i32) -> i64
// CHECK:             %[[VAL_30:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_29]])  : (!fir.ref<!fir.array<10x!fir.type<t{p:!fir.boxproc<() -> i32>}>>>, i64) -> !fir.ref<!fir.type<t{p:!fir.boxproc<() -> i32>}>>
// CHECK:             %[[VAL_31:.*]] = hlfir.designate %[[VAL_30]]{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<t{p:!fir.boxproc<() -> i32>}>>) -> !fir.ref<!fir.boxproc<() -> i32>>
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (!fir.ref<!fir.boxproc<() -> i32>>) -> i64
// CHECK:             fir.store %[[VAL_32]] to %[[VAL_1]] : !fir.ref<i64>
// CHECK:             %[[VAL_33:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<i64>) -> !fir.box<i64>
// CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (!fir.box<i64>) -> !fir.box<none>
// CHECK:             fir.call @_FortranAPushValue(%[[VAL_20]], %[[VAL_34]]) : (!fir.llvm_ptr<i8>, !fir.box<none>) -> ()
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
// CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
// CHECK:           %[[VAL_37:.*]] = arith.constant 1 : index
// CHECK:           fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK:           fir.do_loop %[[VAL_38:.*]] = %[[VAL_35]] to %[[VAL_36]] step %[[VAL_37]] {
// CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (index) -> i64
// CHECK:             %[[VAL_40:.*]] = fir.load %[[VAL_3]] : !fir.ref<i64>
// CHECK:             %[[VAL_41:.*]] = arith.addi %[[VAL_40]], %[[VAL_15]] : i64
// CHECK:             fir.store %[[VAL_41]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK:             %[[VAL_42:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<i64>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             fir.call @_FortranAValueAt(%[[VAL_20]], %[[VAL_40]], %[[VAL_42]]) : (!fir.llvm_ptr<i8>, i64, !fir.ref<!fir.box<none>>) -> ()
// CHECK:             %[[VAL_43:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<i64>>>
// CHECK:             %[[VAL_44:.*]] = fir.box_addr %[[VAL_43]] : (!fir.box<!fir.heap<i64>>) -> !fir.heap<i64>
// CHECK:             %[[VAL_45:.*]] = fir.load %[[VAL_44]] : !fir.heap<i64>
// CHECK:             %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (i64) -> !fir.ref<!fir.boxproc<() -> i32>>
// CHECK:             %[[VAL_47:.*]] = fir.address_of(@f1) : () -> i32
// CHECK:             %[[VAL_48:.*]] = fir.emboxproc %[[VAL_47]] : (() -> i32) -> !fir.boxproc<() -> ()>
// CHECK:             %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (!fir.boxproc<() -> ()>) -> !fir.boxproc<() -> i32>
// CHECK:             fir.store %[[VAL_49]] to %[[VAL_46]] : !fir.ref<!fir.boxproc<() -> i32>>
// CHECK:           }
// CHECK:           fir.call @_FortranADestroyValueStack(%[[VAL_20]]) : (!fir.llvm_ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }

func.func private @f1() -> i32 attributes {fir.proc_attrs = #fir.proc_attrs<pure>}
