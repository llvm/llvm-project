// Test code generation of hlfir.region_assign representing pointer
// assignments inside FORALL.

// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

!t = !fir.type<t{i:i64}>
!ptr_wrapper = !fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>

func.func @test_no_conflict(%n: i64, %arg1: !fir.box<!fir.array<?x!ptr_wrapper>>, %arg2: !fir.ref<!t> ) {
  %c1 = arith.constant 1 : i64
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "a"} : (!fir.box<!fir.array<?x!ptr_wrapper>>, !fir.dscope) -> (!fir.box<!fir.array<?x!ptr_wrapper>>, !fir.box<!fir.array<?x!ptr_wrapper>>)
  %3:2 = hlfir.declare %arg2 dummy_scope %0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "somet"} : (!fir.ref<!t>, !fir.dscope) -> (!fir.ref<!t>, !fir.ref<!t>)
  hlfir.forall lb {
    hlfir.yield %c1 : i64
  } ub {
    hlfir.yield %n : i64
  }  (%arg3: i64) {
    hlfir.region_assign {
      %5 = fir.embox %3#0 : (!fir.ref<!t>) -> !fir.box<!fir.ptr<!t>>
      hlfir.yield %5 : !fir.box<!fir.ptr<!t>>
    } to {
      %6 = hlfir.designate %1#0 (%arg3)  : (!fir.box<!fir.array<?x!ptr_wrapper>>, i64) -> !fir.ref<!ptr_wrapper>
      %7 = hlfir.designate %6{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!ptr_wrapper>) -> !fir.ref<!fir.box<!fir.ptr<!t>>>
      hlfir.yield %7 : !fir.ref<!fir.box<!fir.ptr<!t>>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_no_conflict(
// CHECK-SAME:                                %[[VAL_0:.*]]: i64,
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare{{.*}}"a"
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare{{.*}}"somet"
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_0]] : (i64) -> index
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_10:.*]] = %[[VAL_7]] to %[[VAL_8]] step %[[VAL_9]] {
// CHECK:             %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (index) -> i64
// CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_11]])  : (!fir.box<!fir.array<?x!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>>, i64) -> !fir.ref<!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>
// CHECK:             %[[VAL_13:.*]] = hlfir.designate %[[VAL_12]]{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>
// CHECK:             %[[VAL_14:.*]] = fir.embox %[[VAL_6]]#0 : (!fir.ref<!fir.type<t{i:i64}>>) -> !fir.box<!fir.ptr<!fir.type<t{i:i64}>>>
// CHECK:             fir.store %[[VAL_14]] to %[[VAL_13]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @test_need_to_save_rhs(%n: i64, %arg1: !fir.box<!fir.array<?x!ptr_wrapper>> ) {
  %c1 = arith.constant 1 : i64
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "a"} : (!fir.box<!fir.array<?x!ptr_wrapper>>, !fir.dscope) -> (!fir.box<!fir.array<?x!ptr_wrapper>>, !fir.box<!fir.array<?x!ptr_wrapper>>)
  hlfir.forall lb {
    hlfir.yield %c1 : i64
  } ub {
    hlfir.yield %n : i64
  }  (%arg2: i64) {
    hlfir.region_assign {
      %5 = arith.addi %n, %c1: i64
      %6 = arith.subi %5, %arg2 : i64
      %8 = hlfir.designate %1#0 (%6)  : (!fir.box<!fir.array<?x!ptr_wrapper>>, i64) -> !fir.ref<!ptr_wrapper>
      %9 = hlfir.designate %8{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!ptr_wrapper>) -> !fir.ref<!fir.box<!fir.ptr<!t>>>
      %10 = fir.load %9 : !fir.ref<!fir.box<!fir.ptr<!t>>>
      hlfir.yield %10 : !fir.box<!fir.ptr<!t>>
    } to {
      %5 = hlfir.designate %1#0 (%arg2)  : (!fir.box<!fir.array<?x!ptr_wrapper>>, i64) -> !fir.ref<!ptr_wrapper>
      %6 = hlfir.designate %5{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!ptr_wrapper>) -> !fir.ref<!fir.box<!fir.ptr<!t>>>
      hlfir.yield %6 : !fir.ref<!fir.box<!fir.ptr<!t>>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_need_to_save_rhs(
// CHECK-SAME:                                %[[VAL_0:.*]]: i64,
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.type<t{i:i64}>>>
// CHECK:           %[[VAL_3:.*]] = fir.alloca i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare{{.*}}"a"
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_0]] : (i64) -> index
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : i64
// CHECK:           fir.store %[[VAL_10]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK:           %[[VAL_16:.*]] = fir.call @_FortranACreateDescriptorStack(
// CHECK:           fir.do_loop %[[VAL_17:.*]] = %[[VAL_7]] to %[[VAL_8]] step %[[VAL_9]] {
// CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
// CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_0]], %[[VAL_4]] : i64
// CHECK:             %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_18]] : i64
// CHECK:             %[[VAL_21:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_20]])  : (!fir.box<!fir.array<?x!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>>, i64) -> !fir.ref<!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>
// CHECK:             %[[VAL_22:.*]] = hlfir.designate %[[VAL_21]]{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>
// CHECK:             %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>
// CHECK:             %[[VAL_24:.*]] = fir.box_addr %[[VAL_23]] : (!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>) -> !fir.ptr<!fir.type<t{i:i64}>>
// CHECK:             %[[VAL_25:.*]] = fir.embox %[[VAL_24]] : (!fir.ptr<!fir.type<t{i:i64}>>) -> !fir.box<!fir.type<t{i:i64}>>
// CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.box<!fir.type<t{i:i64}>>) -> !fir.box<none>
// CHECK:             fir.call @_FortranAPushDescriptor(%[[VAL_16]], %[[VAL_26]]) : (!fir.llvm_ptr<i8>, !fir.box<none>) -> ()
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
// CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_0]] : (i64) -> index
// CHECK:           %[[VAL_29:.*]] = arith.constant 1 : index
// CHECK:           fir.store %[[VAL_10]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK:           fir.do_loop %[[VAL_30:.*]] = %[[VAL_27]] to %[[VAL_28]] step %[[VAL_29]] {
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
// CHECK:             %[[VAL_32:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_31]])  : (!fir.box<!fir.array<?x!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>>, i64) -> !fir.ref<!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>
// CHECK:             %[[VAL_33:.*]] = hlfir.designate %[[VAL_32]]{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>
// CHECK:             %[[VAL_34:.*]] = fir.load %[[VAL_3]] : !fir.ref<i64>
// CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_34]], %[[VAL_11]] : i64
// CHECK:             fir.store %[[VAL_35]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             fir.call @_FortranADescriptorAt(%[[VAL_16]], %[[VAL_34]], %[[VAL_36]]) : (!fir.llvm_ptr<i8>, i64, !fir.ref<!fir.box<none>>) -> ()
// CHECK:             %[[VAL_37:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>
// CHECK:             fir.store %[[VAL_37]] to %[[VAL_33]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>
// CHECK:           }
// CHECK:           fir.call @_FortranADestroyDescriptorStack(%[[VAL_16]]) : (!fir.llvm_ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }

func.func @test_need_to_save_lhs(%n: i64, %arg1: !fir.box<!fir.array<?x!ptr_wrapper>>, %arg2: !fir.ref<!t> ) {
  %c1 = arith.constant 1 : i64
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "a"} : (!fir.box<!fir.array<?x!ptr_wrapper>>, !fir.dscope) -> (!fir.box<!fir.array<?x!ptr_wrapper>>, !fir.box<!fir.array<?x!ptr_wrapper>>)
  %3:2 = hlfir.declare %arg2 dummy_scope %0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "somet"} : (!fir.ref<!t>, !fir.dscope) -> (!fir.ref<!t>, !fir.ref<!t>)
  hlfir.forall lb {
    hlfir.yield %c1 : i64
  } ub {
    hlfir.yield %n : i64
  }  (%arg3: i64) {
    hlfir.region_assign {
      %5 = fir.embox %3#0 : (!fir.ref<!t>) -> !fir.box<!fir.ptr<!t>>
      hlfir.yield %5 : !fir.box<!fir.ptr<!t>>
    } to {
      %6 = arith.addi %n, %c1 : i64
      %7 = arith.subi %6, %arg3 : i64
      %9 = hlfir.designate %1#0 (%7)  : (!fir.box<!fir.array<?x!ptr_wrapper>>, i64) -> !fir.ref<!ptr_wrapper>
      %10 = hlfir.designate %9{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!ptr_wrapper>) -> !fir.ref<!fir.box<!fir.ptr<!t>>>
      %11 = fir.load %10 : !fir.ref<!fir.box<!fir.ptr<!t>>>
      %13 = hlfir.designate %11{"i"}   : (!fir.box<!fir.ptr<!t>>) -> !fir.ref<i64>
      %14 = fir.load %13 : !fir.ref<i64>
      %16 = hlfir.designate %1#0 (%14)  : (!fir.box<!fir.array<?x!ptr_wrapper>>, i64) -> !fir.ref<!ptr_wrapper>
      %17 = hlfir.designate %16{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!ptr_wrapper>) -> !fir.ref<!fir.box<!fir.ptr<!t>>>
      hlfir.yield %17 : !fir.ref<!fir.box<!fir.ptr<!t>>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_need_to_save_lhs(
// CHECK-SAME:                                %[[VAL_0:.*]]: i64,
// CHECK:           %[[VAL_3:.*]] = fir.alloca i64
// CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.heap<i64>>
// CHECK:           %[[VAL_5:.*]] = fir.alloca i64
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare{{.*}}"a"
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare{{.*}}"somet"
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_0]] : (i64) -> index
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : i64
// CHECK:           fir.store %[[VAL_13]] to %[[VAL_5]] : !fir.ref<i64>
// CHECK:           %[[VAL_19:.*]] = fir.call @_FortranACreateValueStack(
// CHECK:           fir.do_loop %[[VAL_20:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_12]] {
// CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (index) -> i64
// CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_0]], %[[VAL_6]] : i64
// CHECK:             %[[VAL_23:.*]] = arith.subi %[[VAL_22]], %[[VAL_21]] : i64
// CHECK:             %[[VAL_24:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_23]])  : (!fir.box<!fir.array<?x!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>>, i64) -> !fir.ref<!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>
// CHECK:             %[[VAL_25:.*]] = hlfir.designate %[[VAL_24]]{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>
// CHECK:             %[[VAL_26:.*]] = fir.load %[[VAL_25]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>
// CHECK:             %[[VAL_27:.*]] = hlfir.designate %[[VAL_26]]{"i"}   : (!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>) -> !fir.ref<i64>
// CHECK:             %[[VAL_28:.*]] = fir.load %[[VAL_27]] : !fir.ref<i64>
// CHECK:             %[[VAL_29:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_28]])  : (!fir.box<!fir.array<?x!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>>, i64) -> !fir.ref<!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>
// CHECK:             %[[VAL_30:.*]] = hlfir.designate %[[VAL_29]]{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<ptr_wrapper{p:!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>) -> i64
// CHECK:             fir.store %[[VAL_31]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK:             %[[VAL_32:.*]] = fir.embox %[[VAL_3]] : (!fir.ref<i64>) -> !fir.box<i64>
// CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (!fir.box<i64>) -> !fir.box<none>
// CHECK:             fir.call @_FortranAPushValue(%[[VAL_19]], %[[VAL_33]]) : (!fir.llvm_ptr<i8>, !fir.box<none>) -> ()
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
// CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_0]] : (i64) -> index
// CHECK:           %[[VAL_36:.*]] = arith.constant 1 : index
// CHECK:           fir.store %[[VAL_13]] to %[[VAL_5]] : !fir.ref<i64>
// CHECK:           fir.do_loop %[[VAL_37:.*]] = %[[VAL_34]] to %[[VAL_35]] step %[[VAL_36]] {
// CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (index) -> i64
// CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_5]] : !fir.ref<i64>
// CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_39]], %[[VAL_14]] : i64
// CHECK:             fir.store %[[VAL_40]] to %[[VAL_5]] : !fir.ref<i64>
// CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.box<!fir.heap<i64>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             fir.call @_FortranAValueAt(%[[VAL_19]], %[[VAL_39]], %[[VAL_41]]) : (!fir.llvm_ptr<i8>, i64, !fir.ref<!fir.box<none>>) -> ()
// CHECK:             %[[VAL_42:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.box<!fir.heap<i64>>>
// CHECK:             %[[VAL_43:.*]] = fir.box_addr %[[VAL_42]] : (!fir.box<!fir.heap<i64>>) -> !fir.heap<i64>
// CHECK:             %[[VAL_44:.*]] = fir.load %[[VAL_43]] : !fir.heap<i64>
// CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i64) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>
// CHECK:             %[[VAL_46:.*]] = fir.embox %[[VAL_9]]#0 : (!fir.ref<!fir.type<t{i:i64}>>) -> !fir.box<!fir.ptr<!fir.type<t{i:i64}>>>
// CHECK:             fir.store %[[VAL_46]] to %[[VAL_45]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<t{i:i64}>>>>
// CHECK:           }
// CHECK:           fir.call @_FortranADestroyValueStack(%[[VAL_19]]) : (!fir.llvm_ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }
