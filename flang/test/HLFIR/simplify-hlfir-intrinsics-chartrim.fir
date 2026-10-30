// RUN: fir-opt %s --split-input-file --simplify-hlfir-intrinsics | FileCheck %s

// TRIM of a character variable is rewritten to an inline LEN_TRIM and a copy
// of the substring x(1:LEN_TRIM(x)) into a temporary, removing the runtime
// call. A constant-length operand within the small-array threshold gets a
// constant-size stack temporary hoisted to the entry block of the function or
// of the enclosing OpenMP region, as long as the TRIM temporaries of that
// block together stay within the threshold (or a smaller configured stack
// limit); other operands get a heap temporary, even under -fstack-arrays,
// except in device code where the hlfir.char_trim is kept. Expression,
// polymorphic and volatile operands, and operands that cannot be traced back
// to a declaration, also keep the hlfir.char_trim.

// Sizes are taken from the data layout, as for the other allocation policy
// users.
module attributes {llvm.data_layout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"} {

//  subroutine dummy(x)
//    character(*) :: x
//    call sub(trim(x))
//  end subroutine
func.func @_QPdummy(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %2:2 = hlfir.declare %1#0 typeparams %1#1 dummy_scope %0 uniq_name("_QFdummyEx") : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %3 = hlfir.char_trim %2#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  %len = hlfir.get_length %3 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %4:3 = hlfir.associate %3 typeparams %len {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub(%4#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %3 : !hlfir.expr<!fir.char<1,?>>
  return
}
func.func private @_QPsub(!fir.boxchar<1>)
// CHECK-LABEL:   func.func @_QPdummy(
// CHECK-NOT:       hlfir.char_trim
// CHECK:           %[[X:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[LEN:.*]] dummy_scope
// CHECK:           %[[LAST:.*]] = arith.subi %[[LEN]], %c1{{.*}} : index
// CHECK:           %[[SCAN:.*]]:2 = fir.iterate_while (%[[I:.*]] = %[[LAST]] to %c0{{.*}} step %c-1{{.*}}) and (%{{.*}} = %true{{.*}}) iter_args(%{{.*}} = %[[LAST]]) -> (index) {
// CHECK:             %[[C:.*]] = fir.load %{{.*}} : !fir.ref<i8>
// CHECK:             %[[BLANK:.*]] = arith.cmpi eq, %[[C]], %c32{{.*}} : i8
// CHECK:             fir.result %[[BLANK]], %[[I]] : i1, index
// CHECK:           }
// CHECK:           %[[LENP1:.*]] = arith.addi %[[SCAN]]#1, %c1{{.*}} : index
// CHECK:           %[[LENTRIM:.*]] = arith.select %[[SCAN]]#0, %c0{{.*}}, %[[LENP1]] : index
// CHECK:           %[[SUBSTR:.*]] = hlfir.designate %[[X]]#0  substr %c1{{.*}}, %[[LENTRIM]]  typeparams %[[LENTRIM]] : (!fir.boxchar<1>, index, index, index) -> !fir.boxchar<1>
// CHECK:           %[[MEM:.*]] = fir.allocmem !fir.char<1,?>(%[[LENTRIM]] : index) <{bindc_name = ".tmp.trim", uniq_name = ""}>
// CHECK:           %[[T:.*]]:2 = hlfir.declare %[[MEM]] typeparams %[[LENTRIM]] uniq_name(".tmp.trim") : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
// CHECK:           hlfir.assign %[[SUBSTR]] to %[[T]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           %[[RES:.*]] = hlfir.as_expr %[[T]]#0 move %true{{.*}} : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.char_trim
// CHECK:           hlfir.associate %[[RES]]
// CHECK:           hlfir.destroy %[[RES]]

//  subroutine concat(x, y, r)
//    character(*) :: x, y, r
//    r = trim(x) // trim(y)
//  end subroutine
// Both operands are rewritten; the concatenation consumes the resulting
// hlfir.expr values unchanged.
func.func @_QPconcat(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "y"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "r"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %2:2 = hlfir.declare %1#0 typeparams %1#1 dummy_scope %0 uniq_name("_QFconcatEx") : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %3:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %4:2 = hlfir.declare %3#0 typeparams %3#1 dummy_scope %0 uniq_name("_QFconcatEy") : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %5:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %6:2 = hlfir.declare %5#0 typeparams %5#1 dummy_scope %0 uniq_name("_QFconcatEr") : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %7 = hlfir.char_trim %2#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  %8 = hlfir.get_length %7 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %9 = hlfir.char_trim %4#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  %10 = hlfir.get_length %9 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %11 = arith.addi %8, %10 : index
  %12 = hlfir.concat %7, %9 len %11 : (!hlfir.expr<!fir.char<1,?>>, !hlfir.expr<!fir.char<1,?>>, index) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %12 to %6#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %12 : !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %9 : !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %7 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPconcat(
// CHECK-NOT:       hlfir.char_trim
// CHECK:           %[[X:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFconcatEx")
// CHECK:           %[[Y:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFconcatEy")
// CHECK:           %[[SX:.*]] = hlfir.designate %[[X]]#0  substr %c1{{.*}}, %[[LX:.*]]  typeparams %[[LX]] : (!fir.boxchar<1>, index, index, index) -> !fir.boxchar<1>
// CHECK:           %[[TX:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[LX]] uniq_name(".tmp.trim") : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
// CHECK:           hlfir.assign %[[SX]] to %[[TX]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           %[[VX:.*]] = hlfir.as_expr %[[TX]]#0 move %true{{.*}} : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
// CHECK:           %[[SY:.*]] = hlfir.designate %[[Y]]#0  substr %c1{{.*}}, %[[LY:.*]]  typeparams %[[LY]] : (!fir.boxchar<1>, index, index, index) -> !fir.boxchar<1>
// CHECK:           %[[TY:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[LY]] uniq_name(".tmp.trim") : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
// CHECK:           hlfir.assign %[[SY]] to %[[TY]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           %[[VY:.*]] = hlfir.as_expr %[[TY]]#0 move %true{{.*}} : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
// CHECK:           hlfir.concat %[[VX]], %[[VY]] len %{{.*}} : (!hlfir.expr<!fir.char<1,?>>, !hlfir.expr<!fir.char<1,?>>, index) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.char_trim

//  subroutine fixed()
//    character(8) :: x
//    call sub(trim(x))
//  end subroutine
// The operand has a constant length within the small-array threshold: the
// temporary is a constant-size stack allocation hoisted to the entry block.
func.func @_QPfixed() {
  %c8 = arith.constant 8 : index
  %0 = fir.alloca !fir.char<1,8> {bindc_name = "x", uniq_name = "_QFfixedEx"}
  %1:2 = hlfir.declare %0 typeparams %c8 uniq_name("_QFfixedEx") : (!fir.ref<!fir.char<1,8>>, index) -> (!fir.ref<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>)
  %2 = hlfir.char_trim %1#0 : (!fir.ref<!fir.char<1,8>>) -> !hlfir.expr<!fir.char<1,?>>
  %3 = hlfir.get_length %2 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %4:3 = hlfir.associate %2 typeparams %3 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub(%4#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %2 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPfixed() {
// CHECK-NOT:       hlfir.char_trim
// CHECK:           %[[TMP:.*]] = fir.alloca !fir.char<1,8> <{bindc_name = ".tmp.trim"}>
// CHECK:           %[[X:.*]]:2 = hlfir.declare %{{.*}} typeparams %c8{{.*}} uniq_name("_QFfixedEx")
// CHECK:           fir.iterate_while (%{{.*}} = %c7{{.*}} to %c0{{.*}} step %c-1{{.*}})
// CHECK:           %[[LENTRIM:.*]] = arith.select %{{.*}}, %c0{{.*}}, %{{.*}} : index
// CHECK:           %[[SUBSTR:.*]] = hlfir.designate %[[X]]#0  substr %c1{{.*}}, %[[LENTRIM]]  typeparams %[[LENTRIM]] : (!fir.ref<!fir.char<1,8>>, index, index, index) -> !fir.boxchar<1>
// CHECK-NOT:       fir.allocmem
// CHECK:           %[[REF:.*]] = fir.convert %[[TMP]] : (!fir.ref<!fir.char<1,8>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[T:.*]]:2 = hlfir.declare %[[REF]] typeparams %[[LENTRIM]] uniq_name(".tmp.trim") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           hlfir.assign %[[SUBSTR]] to %[[T]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           hlfir.as_expr %[[T]]#0 move %false{{.*}} : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.char_trim

//  subroutine two()
//    character(8) :: x
//    call sub2(trim(x), trim(x))
//  end subroutine
// Two sites in one statement are live at the same time and get two
// temporaries.
func.func @_QPtwo() {
  %c8 = arith.constant 8 : index
  %0 = fir.alloca !fir.char<1,8> {bindc_name = "x", uniq_name = "_QFtwoEx"}
  %1:2 = hlfir.declare %0 typeparams %c8 uniq_name("_QFtwoEx") : (!fir.ref<!fir.char<1,8>>, index) -> (!fir.ref<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>)
  %2 = hlfir.char_trim %1#0 : (!fir.ref<!fir.char<1,8>>) -> !hlfir.expr<!fir.char<1,?>>
  %3 = hlfir.get_length %2 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %4:3 = hlfir.associate %2 typeparams %3 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  %5 = hlfir.char_trim %1#0 : (!fir.ref<!fir.char<1,8>>) -> !hlfir.expr<!fir.char<1,?>>
  %6 = hlfir.get_length %5 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %7:3 = hlfir.associate %5 typeparams %6 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub2(%4#0, %7#0) : (!fir.boxchar<1>, !fir.boxchar<1>) -> ()
  hlfir.end_associate %7#1, %7#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %5 : !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %2 : !hlfir.expr<!fir.char<1,?>>
  return
}
func.func private @_QPsub2(!fir.boxchar<1>, !fir.boxchar<1>)
// CHECK-LABEL:   func.func @_QPtwo() {
// CHECK:           %[[TA:.*]] = fir.alloca !fir.char<1,8> <{bindc_name = ".tmp.trim"}>
// CHECK:           %[[TB:.*]] = fir.alloca !fir.char<1,8> <{bindc_name = ".tmp.trim"}>
// CHECK-NOT:       .tmp.trim
// CHECK:           hlfir.declare %{{.*}} uniq_name("_QFtwoEx")
// CHECK-NOT:       hlfir.char_trim
// CHECK-DAG:       fir.convert %[[TA]] : (!fir.ref<!fir.char<1,8>>) -> !fir.ref<!fir.char<1,?>>
// CHECK-DAG:       fir.convert %[[TB]] : (!fir.ref<!fir.char<1,8>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           fir.call @_QPsub2(
// CHECK-NOT:       hlfir.char_trim

//  subroutine budget()
//    character(600) :: x
//    call sub2(trim(x), trim(x))
//  end subroutine
// Two sites that fit the threshold individually but not together: the
// second temporary goes on the heap.
func.func @_QPbudget() {
  %c600 = arith.constant 600 : index
  %0 = fir.alloca !fir.char<1,600> {bindc_name = "x", uniq_name = "_QFbudgetEx"}
  %1:2 = hlfir.declare %0 typeparams %c600 uniq_name("_QFbudgetEx") : (!fir.ref<!fir.char<1,600>>, index) -> (!fir.ref<!fir.char<1,600>>, !fir.ref<!fir.char<1,600>>)
  %2 = hlfir.char_trim %1#0 : (!fir.ref<!fir.char<1,600>>) -> !hlfir.expr<!fir.char<1,?>>
  %3 = hlfir.get_length %2 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %4:3 = hlfir.associate %2 typeparams %3 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  %5 = hlfir.char_trim %1#0 : (!fir.ref<!fir.char<1,600>>) -> !hlfir.expr<!fir.char<1,?>>
  %6 = hlfir.get_length %5 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %7:3 = hlfir.associate %5 typeparams %6 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub2(%4#0, %7#0) : (!fir.boxchar<1>, !fir.boxchar<1>) -> ()
  hlfir.end_associate %7#1, %7#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %5 : !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %2 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPbudget() {
// CHECK:           fir.alloca !fir.char<1,600> <{bindc_name = ".tmp.trim"}>
// CHECK-NOT:       .tmp.trim
// CHECK:           hlfir.declare %{{.*}} uniq_name("_QFbudgetEx")
// CHECK-NOT:       hlfir.char_trim
// CHECK:           fir.allocmem !fir.char<1,?>(%{{.*}} : index) <{bindc_name = ".tmp.trim", uniq_name = ""}>
// CHECK-NOT:       fir.allocmem
// CHECK:           fir.call @_QPsub2(
// CHECK-NOT:       hlfir.char_trim

//  subroutine big()
//    character(2048) :: x
//    call sub(trim(x))
//  end subroutine
// The operand has a constant length above the small-array threshold: the
// temporary goes on the heap.
func.func @_QPbig() {
  %c2048 = arith.constant 2048 : index
  %0 = fir.alloca !fir.char<1,2048> {bindc_name = "x", uniq_name = "_QFbigEx"}
  %1:2 = hlfir.declare %0 typeparams %c2048 uniq_name("_QFbigEx") : (!fir.ref<!fir.char<1,2048>>, index) -> (!fir.ref<!fir.char<1,2048>>, !fir.ref<!fir.char<1,2048>>)
  %2 = hlfir.char_trim %1#0 : (!fir.ref<!fir.char<1,2048>>) -> !hlfir.expr<!fir.char<1,?>>
  %3 = hlfir.get_length %2 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %4:3 = hlfir.associate %2 typeparams %3 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub(%4#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %2 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPbig() {
// CHECK-NOT:       hlfir.char_trim
// CHECK-NOT:       fir.alloca !fir.char<1,2048> <{bindc_name = ".tmp.trim"}>
// CHECK:           %[[SUBSTR:.*]] = hlfir.designate %{{.*}}#0  substr %c1{{.*}}, %[[LENTRIM:.*]]  typeparams %[[LENTRIM]] : (!fir.ref<!fir.char<1,2048>>, index, index, index) -> !fir.boxchar<1>
// CHECK:           %[[MEM:.*]] = fir.allocmem !fir.char<1,?>(%[[LENTRIM]] : index) <{bindc_name = ".tmp.trim", uniq_name = ""}>
// CHECK:           %[[T:.*]]:2 = hlfir.declare %[[MEM]] typeparams %[[LENTRIM]] uniq_name(".tmp.trim") : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
// CHECK:           hlfir.assign %[[SUBSTR]] to %[[T]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           hlfir.as_expr %[[T]]#0 move %true{{.*}} : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.char_trim

//  subroutine big_stack_arrays()  ! -fstack-arrays
//    character(2048) :: x
//    call sub(trim(x))
//  end subroutine
// -fstack-arrays does not lift the small-array threshold for this temporary:
// it still goes on the heap.
func.func @_QPbig_stack_arrays() attributes {fir.allocation_policy = #fir.allocation_policy<stack_arrays = true, small_array_threshold = 1024, total_stack_limit = 4194304>} {
  %c2048 = arith.constant 2048 : index
  %0 = fir.alloca !fir.char<1,2048> {bindc_name = "x", uniq_name = "_QFbig_stack_arraysEx"}
  %1:2 = hlfir.declare %0 typeparams %c2048 uniq_name("_QFbig_stack_arraysEx") : (!fir.ref<!fir.char<1,2048>>, index) -> (!fir.ref<!fir.char<1,2048>>, !fir.ref<!fir.char<1,2048>>)
  %2 = hlfir.char_trim %1#0 : (!fir.ref<!fir.char<1,2048>>) -> !hlfir.expr<!fir.char<1,?>>
  %3 = hlfir.get_length %2 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %4:3 = hlfir.associate %2 typeparams %3 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub(%4#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %2 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPbig_stack_arrays() attributes {fir.allocation_policy = #fir.allocation_policy<stack_arrays = true, small_array_threshold = 1024, total_stack_limit = 4194304>} {
// CHECK-NOT:       hlfir.char_trim
// CHECK-NOT:       fir.alloca !fir.char<1,2048> <{bindc_name = ".tmp.trim"}>
// CHECK:           %[[MEM:.*]] = fir.allocmem !fir.char<1,?>(%[[LENTRIM:.*]] : index) <{bindc_name = ".tmp.trim", uniq_name = ""}>
// CHECK:           %[[T:.*]]:2 = hlfir.declare %[[MEM]] typeparams %[[LENTRIM]] uniq_name(".tmp.trim") : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
// CHECK:           hlfir.as_expr %[[T]]#0 move %true{{.*}} : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.char_trim

//  subroutine small_limit()  ! total stack limit below the threshold
//    character(600) :: x
//    call sub(trim(x))
//  end subroutine
// The cap never raises a smaller configured stack limit: the temporary goes
// on the heap.
func.func @_QPsmall_limit() attributes {fir.allocation_policy = #fir.allocation_policy<stack_arrays = false, small_array_threshold = 1024, total_stack_limit = 512>} {
  %c600 = arith.constant 600 : index
  %0 = fir.alloca !fir.char<1,600> {bindc_name = "x", uniq_name = "_QFsmall_limitEx"}
  %1:2 = hlfir.declare %0 typeparams %c600 uniq_name("_QFsmall_limitEx") : (!fir.ref<!fir.char<1,600>>, index) -> (!fir.ref<!fir.char<1,600>>, !fir.ref<!fir.char<1,600>>)
  %2 = hlfir.char_trim %1#0 : (!fir.ref<!fir.char<1,600>>) -> !hlfir.expr<!fir.char<1,?>>
  %3 = hlfir.get_length %2 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %4:3 = hlfir.associate %2 typeparams %3 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub(%4#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %2 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPsmall_limit() attributes
// CHECK-NOT:       hlfir.char_trim
// CHECK-NOT:       .tmp.trim
// CHECK:           hlfir.declare %{{.*}} uniq_name("_QFsmall_limitEx")
// CHECK:           fir.allocmem !fir.char<1,?>(%{{.*}} : index) <{bindc_name = ".tmp.trim", uniq_name = ""}>
// CHECK-NOT:       hlfir.char_trim

//  subroutine kind4(x)
//    character(*, kind=4) :: x
//    call sub4(trim(x))
//  end subroutine
func.func @_QPkind4(%arg0: !fir.boxchar<4> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
  %2:2 = hlfir.declare %1#0 typeparams %1#1 dummy_scope %0 uniq_name("_QFkind4Ex") : (!fir.ref<!fir.char<4,?>>, index, !fir.dscope) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
  %3 = hlfir.char_trim %2#0 : (!fir.boxchar<4>) -> !hlfir.expr<!fir.char<4,?>>
  %len = hlfir.get_length %3 : (!hlfir.expr<!fir.char<4,?>>) -> index
  %4:3 = hlfir.associate %3 typeparams %len {adapt.valuebyref} : (!hlfir.expr<!fir.char<4,?>>, index) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>, i1)
  fir.call @_QPsub4(%4#0) : (!fir.boxchar<4>) -> ()
  hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.char<4,?>>, i1
  hlfir.destroy %3 : !hlfir.expr<!fir.char<4,?>>
  return
}
func.func private @_QPsub4(!fir.boxchar<4>)
// CHECK-LABEL:   func.func @_QPkind4(
// CHECK-NOT:       hlfir.char_trim
// CHECK:           %[[C:.*]] = fir.load %{{.*}} : !fir.ref<i32>
// CHECK:           arith.cmpi eq, %[[C]], %c32{{.*}} : i32
// CHECK:           %[[SUBSTR:.*]] = hlfir.designate %{{.*}}#0  substr %c1{{.*}}, %[[LENTRIM:.*]]  typeparams %[[LENTRIM]] : (!fir.boxchar<4>, index, index, index) -> !fir.boxchar<4>
// CHECK:           %[[MEM:.*]] = fir.allocmem !fir.char<4,?>(%[[LENTRIM]] : index) <{bindc_name = ".tmp.trim", uniq_name = ""}>
// CHECK:           %[[T:.*]]:2 = hlfir.declare %[[MEM]] typeparams %[[LENTRIM]] uniq_name(".tmp.trim") : (!fir.heap<!fir.char<4,?>>, index) -> (!fir.boxchar<4>, !fir.heap<!fir.char<4,?>>)
// CHECK:           hlfir.assign %[[SUBSTR]] to %[[T]]#0 : !fir.boxchar<4>, !fir.boxchar<4>
// CHECK:           hlfir.as_expr %[[T]]#0 move %true{{.*}} : (!fir.boxchar<4>, i1) -> !hlfir.expr<!fir.char<4,?>>
// CHECK-NOT:       hlfir.char_trim

//  subroutine ptr(p)
//    character(:), pointer :: p
//    call sub(trim(p))
//  end subroutine
// The operand is a loaded descriptor: the address and length are read from it
// and the substring designates the descriptor directly.
func.func @_QPptr(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>> {fir.bindc_name = "p"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 uniq_name("_QFptrEp") fortran_attrs<pointer> : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>)
  %2 = fir.load %1#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  %3 = hlfir.char_trim %2 : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !hlfir.expr<!fir.char<1,?>>
  %4 = hlfir.get_length %3 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %5:3 = hlfir.associate %3 typeparams %4 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub(%5#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %5#1, %5#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %3 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPptr(
// CHECK-NOT:       hlfir.char_trim
// CHECK:           %[[P:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
// CHECK-DAG:       fir.box_addr %[[P]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
// CHECK-DAG:       %[[LEN:.*]] = fir.box_elesize %[[P]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
// CHECK:           %[[LAST:.*]] = arith.subi %[[LEN]], %c1{{.*}} : index
// CHECK:           fir.iterate_while (%{{.*}} = %[[LAST]] to %c0{{.*}} step %c-1{{.*}})
// CHECK:           %[[LENTRIM:.*]] = arith.select %{{.*}}, %c0{{.*}}, %{{.*}} : index
// CHECK:           %[[SUBSTR:.*]] = hlfir.designate %[[P]]  substr %c1{{.*}}, %[[LENTRIM]]  typeparams %[[LENTRIM]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>, index, index, index) -> !fir.boxchar<1>
// CHECK:           %[[MEM:.*]] = fir.allocmem !fir.char<1,?>(%[[LENTRIM]] : index) <{bindc_name = ".tmp.trim", uniq_name = ""}>
// CHECK:           %[[T:.*]]:2 = hlfir.declare %[[MEM]] typeparams %[[LENTRIM]] uniq_name(".tmp.trim")
// CHECK:           hlfir.assign %[[SUBSTR]] to %[[T]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           hlfir.as_expr %[[T]]#0 move %true{{.*}} : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.char_trim

//  subroutine target_fixed()
//    character(256) :: x
//    !$omp target
//    call sub(trim(x))
//    !$omp end target
//  end subroutine
// In an offload region a constant-size temporary is still a stack allocation,
// placed in the entry block of the region.
func.func @_QPtarget_fixed() {
  %c256 = arith.constant 256 : index
  %0 = fir.alloca !fir.char<1,256> {bindc_name = "x", uniq_name = "_QFtarget_fixedEx"}
  %1:2 = hlfir.declare %0 typeparams %c256 uniq_name("_QFtarget_fixedEx") : (!fir.ref<!fir.char<1,256>>, index) -> (!fir.ref<!fir.char<1,256>>, !fir.ref<!fir.char<1,256>>)
  %2 = omp.map.info var_ptr(%1#1 : !fir.ref<!fir.char<1,256>>, !fir.char<1,256>) map_clauses(implicit, tofrom) capture(ByRef) -> !fir.ref<!fir.char<1,256>>
  omp.target kernel_type(generic) map_entries(%2 -> %arg0 : !fir.ref<!fir.char<1,256>>) {
    %c256_0 = arith.constant 256 : index
    %3:2 = hlfir.declare %arg0 typeparams %c256_0 uniq_name("_QFtarget_fixedEx") : (!fir.ref<!fir.char<1,256>>, index) -> (!fir.ref<!fir.char<1,256>>, !fir.ref<!fir.char<1,256>>)
    %4 = hlfir.char_trim %3#0 : (!fir.ref<!fir.char<1,256>>) -> !hlfir.expr<!fir.char<1,?>>
    %5 = hlfir.get_length %4 : (!hlfir.expr<!fir.char<1,?>>) -> index
    %6:3 = hlfir.associate %4 typeparams %5 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
    fir.call @_QPsub(%6#0) : (!fir.boxchar<1>) -> ()
    hlfir.end_associate %6#1, %6#2 : !fir.ref<!fir.char<1,?>>, i1
    hlfir.destroy %4 : !hlfir.expr<!fir.char<1,?>>
    omp.terminator
  }
  return
}
// CHECK-LABEL:   func.func @_QPtarget_fixed() {
// CHECK-NOT:       .tmp.trim
// CHECK:           omp.target
// CHECK:             %[[TMP:.*]] = fir.alloca !fir.char<1,256> <{bindc_name = ".tmp.trim", pinned}>
// CHECK-NEXT:        hlfir.declare %{{.*}} uniq_name("_QFtarget_fixedEx")
// CHECK-NOT:         hlfir.char_trim
// CHECK-NOT:         fir.allocmem
// CHECK:             %[[REF:.*]] = fir.convert %[[TMP]] : (!fir.ref<!fir.char<1,256>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:             %[[T:.*]]:2 = hlfir.declare %[[REF]] typeparams %{{.*}} uniq_name(".tmp.trim")
// CHECK:             hlfir.as_expr %[[T]]#0 move %false{{.*}} : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
// CHECK:             omp.terminator

//  subroutine par()
//    character(8) :: x
//    !$omp parallel
//    call sub(trim(x))
//    !$omp end parallel
//  end subroutine
// The temporary is placed in the entry block of the parallel region, not of
// the function, so each thread has its own.
func.func @_QPpar() {
  %c8 = arith.constant 8 : index
  %0 = fir.alloca !fir.char<1,8> {bindc_name = "x", uniq_name = "_QFparEx"}
  %1:2 = hlfir.declare %0 typeparams %c8 uniq_name("_QFparEx") : (!fir.ref<!fir.char<1,8>>, index) -> (!fir.ref<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>)
  omp.parallel {
    %2 = hlfir.char_trim %1#0 : (!fir.ref<!fir.char<1,8>>) -> !hlfir.expr<!fir.char<1,?>>
    %3 = hlfir.get_length %2 : (!hlfir.expr<!fir.char<1,?>>) -> index
    %4:3 = hlfir.associate %2 typeparams %3 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
    fir.call @_QPsub(%4#0) : (!fir.boxchar<1>) -> ()
    hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.char<1,?>>, i1
    hlfir.destroy %2 : !hlfir.expr<!fir.char<1,?>>
    omp.terminator
  }
  return
}
// CHECK-LABEL:   func.func @_QPpar() {
// CHECK-NOT:       .tmp.trim
// CHECK:           omp.parallel {
// CHECK:             %[[TMP:.*]] = fir.alloca !fir.char<1,8> <{bindc_name = ".tmp.trim", pinned}>
// CHECK-NOT:         hlfir.char_trim
// CHECK:             fir.iterate_while
// CHECK:             %[[REF:.*]] = fir.convert %[[TMP]] : (!fir.ref<!fir.char<1,8>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:             %[[T:.*]]:2 = hlfir.declare %[[REF]] typeparams %{{.*}} uniq_name(".tmp.trim")
// CHECK:             hlfir.as_expr %[[T]]#0 move %false{{.*}} : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
// CHECK:             omp.terminator

//  subroutine target_big()
//    character(2048) :: x
//    !$omp target
//    call sub(trim(x))
//    !$omp end target
//  end subroutine
// In an offload region a temporary that would go on the heap is not created:
// the runtime call is kept.
func.func @_QPtarget_big() {
  %c2048 = arith.constant 2048 : index
  %0 = fir.alloca !fir.char<1,2048> {bindc_name = "x", uniq_name = "_QFtarget_bigEx"}
  %1:2 = hlfir.declare %0 typeparams %c2048 uniq_name("_QFtarget_bigEx") : (!fir.ref<!fir.char<1,2048>>, index) -> (!fir.ref<!fir.char<1,2048>>, !fir.ref<!fir.char<1,2048>>)
  %2 = omp.map.info var_ptr(%1#1 : !fir.ref<!fir.char<1,2048>>, !fir.char<1,2048>) map_clauses(implicit, tofrom) capture(ByRef) -> !fir.ref<!fir.char<1,2048>>
  omp.target kernel_type(generic) map_entries(%2 -> %arg0 : !fir.ref<!fir.char<1,2048>>) {
    %c2048_0 = arith.constant 2048 : index
    %3:2 = hlfir.declare %arg0 typeparams %c2048_0 uniq_name("_QFtarget_bigEx") : (!fir.ref<!fir.char<1,2048>>, index) -> (!fir.ref<!fir.char<1,2048>>, !fir.ref<!fir.char<1,2048>>)
    %4 = hlfir.char_trim %3#0 : (!fir.ref<!fir.char<1,2048>>) -> !hlfir.expr<!fir.char<1,?>>
    %5 = hlfir.get_length %4 : (!hlfir.expr<!fir.char<1,?>>) -> index
    %6:3 = hlfir.associate %4 typeparams %5 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
    fir.call @_QPsub(%6#0) : (!fir.boxchar<1>) -> ()
    hlfir.end_associate %6#1, %6#2 : !fir.ref<!fir.char<1,?>>, i1
    hlfir.destroy %4 : !hlfir.expr<!fir.char<1,?>>
    omp.terminator
  }
  return
}
// CHECK-LABEL:   func.func @_QPtarget_big() {
// CHECK:           omp.target
// CHECK:             hlfir.char_trim %{{.*}} : (!fir.ref<!fir.char<1,2048>>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:         hlfir.designate
// CHECK:             omp.terminator

//  subroutine target_budget()
//    character(600) :: x
//    !$omp target
//    call sub2(trim(x), trim(x))
//    !$omp end target
//  end subroutine
// Past the shared budget in an offload region, the runtime call is kept.
func.func @_QPtarget_budget() {
  %c600 = arith.constant 600 : index
  %0 = fir.alloca !fir.char<1,600> {bindc_name = "x", uniq_name = "_QFtarget_budgetEx"}
  %1:2 = hlfir.declare %0 typeparams %c600 uniq_name("_QFtarget_budgetEx") : (!fir.ref<!fir.char<1,600>>, index) -> (!fir.ref<!fir.char<1,600>>, !fir.ref<!fir.char<1,600>>)
  %2 = omp.map.info var_ptr(%1#1 : !fir.ref<!fir.char<1,600>>, !fir.char<1,600>) map_clauses(implicit, tofrom) capture(ByRef) -> !fir.ref<!fir.char<1,600>>
  omp.target kernel_type(generic) map_entries(%2 -> %arg0 : !fir.ref<!fir.char<1,600>>) {
    %c600_0 = arith.constant 600 : index
    %3:2 = hlfir.declare %arg0 typeparams %c600_0 uniq_name("_QFtarget_budgetEx") : (!fir.ref<!fir.char<1,600>>, index) -> (!fir.ref<!fir.char<1,600>>, !fir.ref<!fir.char<1,600>>)
    %4 = hlfir.char_trim %3#0 : (!fir.ref<!fir.char<1,600>>) -> !hlfir.expr<!fir.char<1,?>>
    %5 = hlfir.get_length %4 : (!hlfir.expr<!fir.char<1,?>>) -> index
    %6:3 = hlfir.associate %4 typeparams %5 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
    %7 = hlfir.char_trim %3#0 : (!fir.ref<!fir.char<1,600>>) -> !hlfir.expr<!fir.char<1,?>>
    %8 = hlfir.get_length %7 : (!hlfir.expr<!fir.char<1,?>>) -> index
    %9:3 = hlfir.associate %7 typeparams %8 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
    fir.call @_QPsub2(%6#0, %9#0) : (!fir.boxchar<1>, !fir.boxchar<1>) -> ()
    hlfir.end_associate %9#1, %9#2 : !fir.ref<!fir.char<1,?>>, i1
    hlfir.end_associate %6#1, %6#2 : !fir.ref<!fir.char<1,?>>, i1
    hlfir.destroy %7 : !hlfir.expr<!fir.char<1,?>>
    hlfir.destroy %4 : !hlfir.expr<!fir.char<1,?>>
    omp.terminator
  }
  return
}
// CHECK-LABEL:   func.func @_QPtarget_budget() {
// CHECK:           omp.target
// CHECK:             fir.alloca !fir.char<1,600> <{bindc_name = ".tmp.trim", pinned}>
// CHECK-NOT:         .tmp.trim
// CHECK:             hlfir.declare %{{.*}} uniq_name("_QFtarget_budgetEx")
// CHECK:             hlfir.char_trim %{{.*}} : (!fir.ref<!fir.char<1,600>>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:         hlfir.char_trim
// CHECK-NOT:         fir.allocmem
// CHECK:             omp.terminator

//  subroutine nohost(x)
//    !$omp declare target device_type(nohost)
//    character(*) :: x
//    call sub(trim(x))
//  end subroutine
// A device-only procedure: the runtime call is kept.
func.func @_QPnohost(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"}) attributes {omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>} {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %2:2 = hlfir.declare %1#0 typeparams %1#1 dummy_scope %0 uniq_name("_QFnohostEx") : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %3 = hlfir.char_trim %2#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  %len = hlfir.get_length %3 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %4:3 = hlfir.associate %3 typeparams %len {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub(%4#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %3 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPnohost(
// CHECK:           %[[X:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFnohostEx")
// CHECK:           hlfir.char_trim %[[X]]#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.designate

//  subroutine expr(x, y)
//    character(*) :: x, y
//    call sub(trim(x // y))
//  end subroutine
// The operand is an expression with no storage of its own: not rewritten.
func.func @_QPexpr(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "y"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %2:2 = hlfir.declare %1#0 typeparams %1#1 dummy_scope %0 uniq_name("_QFexprEx") : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %3:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %4:2 = hlfir.declare %3#0 typeparams %3#1 dummy_scope %0 uniq_name("_QFexprEy") : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %5 = arith.addi %1#1, %3#1 : index
  %6 = hlfir.concat %2#0, %4#0 len %5 : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  %7 = hlfir.char_trim %6 : (!hlfir.expr<!fir.char<1,?>>) -> !hlfir.expr<!fir.char<1,?>>
  %len = hlfir.get_length %7 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %8:3 = hlfir.associate %7 typeparams %len {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub(%8#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %8#1, %8#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %7 : !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %6 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPexpr(
// CHECK:           %[[CAT:.*]] = hlfir.concat
// CHECK:           hlfir.char_trim %[[CAT]] : (!hlfir.expr<!fir.char<1,?>>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.designate

// No Fortran source: lowering never produces a polymorphic character, but
// the verifier accepts it. Not rewritten.
func.func @_QPpoly(%arg0: !fir.class<!fir.char<1,?>>) {
  %0 = hlfir.char_trim %arg0 : (!fir.class<!fir.char<1,?>>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %0 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPpoly(
// CHECK:           hlfir.char_trim %{{.*}} : (!fir.class<!fir.char<1,?>>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.designate

//  subroutine vol(x)
//    character(8), volatile :: x
//    call sub(trim(x))
//  end subroutine
// The operand type itself is volatile: not rewritten.
func.func @_QPvol(%arg0: !fir.ref<!fir.char<1,8>, volatile> {fir.bindc_name = "x"}) {
  %c8 = arith.constant 8 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 typeparams %c8 dummy_scope %0 uniq_name("_QFvolEx") fortran_attrs<volatile> : (!fir.ref<!fir.char<1,8>, volatile>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,8>, volatile>, !fir.ref<!fir.char<1,8>, volatile>)
  %2 = hlfir.char_trim %1#0 : (!fir.ref<!fir.char<1,8>, volatile>) -> !hlfir.expr<!fir.char<1,?>>
  %3 = hlfir.get_length %2 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %4:3 = hlfir.associate %2 typeparams %3 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub(%4#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %2 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPvol(
// CHECK:           hlfir.char_trim %{{.*}} : (!fir.ref<!fir.char<1,8>, volatile>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.designate

//  subroutine volstar(x)
//    character(*), volatile :: x
//    call sub(trim(x))
//  end subroutine
// The hlfir.declare result is a plain boxchar; volatility is only visible on
// the FIR base and the declaration attributes: not rewritten.
func.func @_QPvolstar(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %2 = fir.volatile_cast %1#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>, volatile>
  %3:2 = hlfir.declare %2 typeparams %1#1 dummy_scope %0 uniq_name("_QFvolstarEx") fortran_attrs<volatile> : (!fir.ref<!fir.char<1,?>, volatile>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>, volatile>)
  %4 = hlfir.char_trim %3#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  %len = hlfir.get_length %4 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %5:3 = hlfir.associate %4 typeparams %len {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub(%5#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %5#1, %5#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %4 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPvolstar(
// CHECK:           %[[X:.*]]:2 = hlfir.declare %{{.*}} fortran_attrs<volatile>
// CHECK:           hlfir.char_trim %[[X]]#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.designate

//  subroutine volsubstr(x)
//    character(*), volatile :: x
//    call sub(trim(x(2:)))
//  end subroutine
// The operand is a boxchar designator; volatility comes from the variable it
// designates: not rewritten.
func.func @_QPvolsubstr(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"}) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %2 = fir.volatile_cast %1#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>, volatile>
  %3:2 = hlfir.declare %2 typeparams %1#1 dummy_scope %0 uniq_name("_QFvolsubstrEx") fortran_attrs<volatile> : (!fir.ref<!fir.char<1,?>, volatile>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>, volatile>)
  %4 = arith.subi %1#1, %c1 : index
  %5 = hlfir.designate %3#0  substr %c2, %1#1  typeparams %4 : (!fir.boxchar<1>, index, index, index) -> !fir.boxchar<1>
  %6 = hlfir.char_trim %5 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  %len = hlfir.get_length %6 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %7:3 = hlfir.associate %6 typeparams %len {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub(%7#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %7#1, %7#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %6 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPvolsubstr(
// CHECK:           %[[S:.*]] = hlfir.designate %{{.*}}#0  substr %c2{{.*}}, %{{.*}}  typeparams %{{.*}} : (!fir.boxchar<1>, index, index, index) -> !fir.boxchar<1>
// CHECK:           hlfir.char_trim %[[S]] : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.designate

//  subroutine volptr(p)
//    character(:), pointer, volatile :: p
//    call sub(trim(p))
//  end subroutine
// The operand is an emboxchar of the box_addr of the loaded descriptor, and
// fir.box_addr drops volatility; it is found on the hlfir.declare: not
// rewritten.
func.func @_QPvolptr(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>, volatile>, volatile> {fir.bindc_name = "p"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 uniq_name("_QFvolptrEp") fortran_attrs<pointer, volatile> : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>, volatile>, volatile>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>, volatile>, volatile>)
  %2 = fir.load %1#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>, volatile>, volatile>
  %3 = fir.box_addr %2 : (!fir.box<!fir.ptr<!fir.char<1,?>>, volatile>) -> !fir.ptr<!fir.char<1,?>>
  %4 = fir.box_elesize %2 : (!fir.box<!fir.ptr<!fir.char<1,?>>, volatile>) -> index
  %5 = fir.emboxchar %3, %4 : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  %6 = hlfir.char_trim %5 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  %len = hlfir.get_length %6 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %7:3 = hlfir.associate %6 typeparams %len {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub(%7#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %7#1, %7#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %6 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPvolptr(
// CHECK:           %[[P:.*]] = fir.emboxchar %{{.*}}, %{{.*}} : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.boxchar<1>
// CHECK:           hlfir.char_trim %[[P]] : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.designate

//  subroutine volalloc(a)
//    character(:), allocatable, volatile :: a
//    call sub(trim(a))
//  end subroutine
// Same as volptr for an ALLOCATABLE: not rewritten.
func.func @_QPvolalloc(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>, volatile>, volatile> {fir.bindc_name = "a"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 uniq_name("_QFvolallocEa") fortran_attrs<allocatable, volatile> : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>, volatile>, volatile>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>, volatile>, volatile>)
  %2 = fir.load %1#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>, volatile>, volatile>
  %3 = fir.box_addr %2 : (!fir.box<!fir.heap<!fir.char<1,?>>, volatile>) -> !fir.heap<!fir.char<1,?>>
  %4 = fir.box_elesize %2 : (!fir.box<!fir.heap<!fir.char<1,?>>, volatile>) -> index
  %5 = fir.emboxchar %3, %4 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  %6 = hlfir.char_trim %5 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  %len = hlfir.get_length %6 : (!hlfir.expr<!fir.char<1,?>>) -> index
  %7:3 = hlfir.associate %6 typeparams %len {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @_QPsub(%7#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %7#1, %7#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %6 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPvolalloc(
// CHECK:           %[[A:.*]] = fir.emboxchar %{{.*}}, %{{.*}} : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
// CHECK:           hlfir.char_trim %[[A]] : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.designate

}

// -----

// In a module compiled for an OpenMP target device, an operand that would need
// a heap temporary keeps the runtime call even outside of a target region
// (e.g. in a declare target procedure).
module attributes {omp.is_target_device = true} {
  func.func @_QPdevice(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"}) {
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %2:2 = hlfir.declare %1#0 typeparams %1#1 dummy_scope %0 uniq_name("_QFdeviceEx") : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %3 = hlfir.char_trim %2#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
    %len = hlfir.get_length %3 : (!hlfir.expr<!fir.char<1,?>>) -> index
    %4:3 = hlfir.associate %3 typeparams %len {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
    fir.call @_QPsub(%4#0) : (!fir.boxchar<1>) -> ()
    hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.char<1,?>>, i1
    hlfir.destroy %3 : !hlfir.expr<!fir.char<1,?>>
    return
  }
  func.func private @_QPsub(!fir.boxchar<1>)
}
// CHECK-LABEL:   func.func @_QPdevice(
// CHECK:           %[[X:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdeviceEx")
// CHECK:           hlfir.char_trim %[[X]]#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.designate
