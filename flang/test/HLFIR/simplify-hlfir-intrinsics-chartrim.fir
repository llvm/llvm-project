// RUN: fir-opt %s --simplify-hlfir-intrinsics | FileCheck %s

// TRIM of a character variable is rewritten to the substring x(1:LEN_TRIM(x))
// with an inline LEN_TRIM, removing the runtime call and its heap allocation.
// Expression and volatile operands keep the hlfir.char_trim, which is lowered
// to the runtime later.

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
// CHECK-SAME:      %[[ARG0:.*]]: !fir.boxchar<1>
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
// CHECK:           %[[VIEW:.*]] = hlfir.as_expr %[[SUBSTR]] : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.char_trim
// CHECK-NOT:       move
// CHECK:           hlfir.associate %[[VIEW]]
// CHECK:           hlfir.destroy %[[VIEW]]

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
// CHECK:           %[[VX:.*]] = hlfir.as_expr %[[SX]] : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK:           %[[SY:.*]] = hlfir.designate %[[Y]]#0  substr %c1{{.*}}, %[[LY:.*]]  typeparams %[[LY]] : (!fir.boxchar<1>, index, index, index) -> !fir.boxchar<1>
// CHECK:           %[[VY:.*]] = hlfir.as_expr %[[SY]] : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK:           hlfir.concat %[[VX]], %[[VY]] len %{{.*}} : (!hlfir.expr<!fir.char<1,?>>, !hlfir.expr<!fir.char<1,?>>, index) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.char_trim
// CHECK-NOT:       move

//  subroutine fixed()
//    character(8) :: x
//    call sub(trim(x))
//  end subroutine
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
// CHECK:           %[[X:.*]]:2 = hlfir.declare %{{.*}} typeparams %c8{{.*}} uniq_name("_QFfixedEx")
// CHECK:           fir.iterate_while (%{{.*}} = %c7{{.*}} to %c0{{.*}} step %c-1{{.*}})
// CHECK:           %[[LENTRIM:.*]] = arith.select %{{.*}}, %c0{{.*}}, %{{.*}} : index
// CHECK:           %[[SUBSTR:.*]] = hlfir.designate %[[X]]#0  substr %c1{{.*}}, %[[LENTRIM]]  typeparams %[[LENTRIM]] : (!fir.ref<!fir.char<1,8>>, index, index, index) -> !fir.boxchar<1>
// CHECK:           hlfir.as_expr %[[SUBSTR]] : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
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
// CHECK:           hlfir.as_expr %[[SUBSTR]] : (!fir.boxchar<4>) -> !hlfir.expr<!fir.char<4,?>>
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
// CHECK:           hlfir.as_expr %[[SUBSTR]] : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
// CHECK-NOT:       hlfir.char_trim

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

//  subroutine vol(x)
//    character(8), volatile :: x
//    call sub(trim(x))
//  end subroutine
// A boxchar cannot carry volatility: not rewritten.
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
