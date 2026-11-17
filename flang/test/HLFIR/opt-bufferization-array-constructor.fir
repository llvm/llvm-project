// RUN: fir-opt --opt-bufferization %s | FileCheck %s

// Build the section and scalar elements directly in the destination.
// CHECK-LABEL: func.func @array_section_constructor(
// CHECK: %[[DEST:.*]]:2 = hlfir.declare %arg1
// CHECK: %[[SOURCE:.*]]:2 = hlfir.declare %arg0
// CHECK-NOT: fir.allocmem
// CHECK: %[[DEST_ADDR:.*]] = fir.convert %[[DEST]]#0 : (!fir.ref<!fir.array<4xf64>>) -> !fir.heap<!fir.array<4xf64>>
// CHECK: %[[TMP:.*]]:2 = hlfir.declare %[[DEST_ADDR]](
// CHECK: %[[SOURCE_SECTION:.*]] = hlfir.designate %[[SOURCE]]#0
// CHECK: fir.do_loop
// CHECK: %[[SOURCE_ELEMENT:.*]] = hlfir.designate %[[SOURCE_SECTION]] (
// CHECK: %[[VALUE:.*]] = fir.load %[[SOURCE_ELEMENT]] : !fir.ref<f64>
// CHECK: %[[SLICE:.*]] = hlfir.designate %[[TMP]]#0
// CHECK: hlfir.assign %[[VALUE]] to %[[SLICE]] : f64, !fir.ref<f64>
// CHECK: }
// CHECK-NOT: hlfir.as_expr
// CHECK-NOT: hlfir.assign
// CHECK-NOT: hlfir.destroy
// CHECK: return
func.func @array_section_constructor(%arg0: !fir.ref<!fir.array<4xf64>> {fir.bindc_name = "stress", fir.fortran_attrs = #fir.var_attrs<intent_in>}, %arg1: !fir.ref<!fir.array<4xf64>> {fir.bindc_name = "sinc", fir.fortran_attrs = #fir.var_attrs<intent_out>}) {
  %true = arith.constant true
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = fir.alloca index
  %1 = fir.dummy_scope : !fir.dscope
  %2 = fir.shape %c4 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %1 arg 2 uniq_name("_QFarray_section_constructorEsinc") fortran_attrs<intent_out> : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)
  %4:2 = hlfir.declare %arg0(%2) dummy_scope %1 arg 1 uniq_name("_QFarray_section_constructorEstress") fortran_attrs<intent_in> : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)
  fir.store %c1 to %0 : !fir.ref<index>
  %5 = fir.allocmem !fir.array<4xf64> <{bindc_name = ".tmp.arrayctor", uniq_name = ""}>
  %6:2 = hlfir.declare %5(%2) uniq_name(".tmp.arrayctor") : (!fir.heap<!fir.array<4xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<4xf64>>, !fir.heap<!fir.array<4xf64>>)
  %7 = hlfir.designate %4#0 (%c1:%c4:%c1)  shape %2 : (!fir.ref<!fir.array<4xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
  fir.do_loop %arg2 = %c1 to %c4 step %c1 {
    %9 = hlfir.designate %7 (%arg2)  : (!fir.ref<!fir.array<4xf64>>, index) -> !fir.ref<f64>
    %10 = fir.load %9 : !fir.ref<f64>
    %11 = fir.load %0 : !fir.ref<index>
    %12 = arith.addi %11, %c1 : index
    fir.store %12 to %0 : !fir.ref<index>
    %13 = hlfir.designate %6#0 (%11)  : (!fir.heap<!fir.array<4xf64>>, index) -> !fir.ref<f64>
    hlfir.assign %10 to %13 : f64, !fir.ref<f64>
  }
  %8 = hlfir.as_expr %6#0 move %true : (!fir.heap<!fir.array<4xf64>>, i1) -> !hlfir.expr<4xf64>
  hlfir.assign %8 to %3#0 : !hlfir.expr<4xf64>, !fir.ref<!fir.array<4xf64>>
  hlfir.destroy %8 : !hlfir.expr<4xf64>
  return
}

// Minimal eligible construction shared by the negative cases below.
// CHECK-LABEL: func.func @simple_constructor(
// CHECK-SAME: %[[DEST:[^:]+]]: !fir.ref<!fir.array<1xf64>>
// CHECK-NOT: fir.allocmem
// CHECK: %[[DEST_ADDR:.*]] = fir.convert %[[DEST]] : (!fir.ref<!fir.array<1xf64>>) -> !fir.heap<!fir.array<1xf64>>
// CHECK: %[[TMP:.*]]:2 = hlfir.declare %[[DEST_ADDR]](
// CHECK: %[[ELEMENT:.*]] = hlfir.designate %[[TMP]]#0
// CHECK: hlfir.assign {{.*}} to %[[ELEMENT]] : f64, !fir.ref<f64>
// CHECK-NOT: hlfir.as_expr
// CHECK-NOT: hlfir.destroy
// CHECK: return
func.func @simple_constructor(%dest: !fir.ref<!fir.array<1xf64>>) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %must_free = arith.constant true
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<1xf64>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1xf64>>, !fir.heap<!fir.array<1xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.heap<!fir.array<1xf64>>, i1) -> !hlfir.expr<1xf64>
  hlfir.assign %expr to %dest : !hlfir.expr<1xf64>, !fir.ref<!fir.array<1xf64>>
  hlfir.destroy %expr : !hlfir.expr<1xf64>
  return
}

// Freeing a distinct allocation between construction and assignment must not
// prevent buffer reuse. Use the same array type so rejection comes from the
// allocation, not a difference in element types.
// CHECK-LABEL: func.func @unrelated_allocation_freed(
// CHECK-SAME: %[[DEST:[^:]+]]: !fir.ref<!fir.array<1xf64>>
// CHECK-NOT: fir.allocmem
// CHECK: %[[OTHER:.*]] = fir.allocmem !fir.array<1xf64>
// CHECK-NEXT: %[[DEST_ADDR:.*]] = fir.convert %[[DEST]] : (!fir.ref<!fir.array<1xf64>>) -> !fir.heap<!fir.array<1xf64>>
// CHECK-NEXT: %[[TMP:.*]]:2 = hlfir.declare %[[DEST_ADDR]](
// CHECK-NEXT: %[[ELEMENT:.*]] = hlfir.designate %[[TMP]]#0
// CHECK-NEXT: hlfir.assign {{.*}} to %[[ELEMENT]] : f64, !fir.ref<f64>
// CHECK-NEXT: fir.freemem %[[OTHER]] : !fir.heap<!fir.array<1xf64>>
// CHECK-NEXT: return
func.func @unrelated_allocation_freed(%dest: !fir.ref<!fir.array<1xf64>>) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %must_free = arith.constant true
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %other = fir.allocmem !fir.array<1xf64>
  %storage = fir.allocmem !fir.array<1xf64>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1xf64>>, !fir.heap<!fir.array<1xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  fir.freemem %other : !fir.heap<!fir.array<1xf64>>
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.heap<!fir.array<1xf64>>, i1) -> !hlfir.expr<1xf64>
  hlfir.assign %expr to %dest : !hlfir.expr<1xf64>, !fir.ref<!fir.array<1xf64>>
  hlfir.destroy %expr : !hlfir.expr<1xf64>
  return
}

// Dest is read so keep the temporary.
// CHECK-LABEL: func.func @self_reference(
// CHECK: fir.allocmem
// CHECK: hlfir.as_expr
// CHECK: hlfir.assign {{.*}} : !hlfir.expr<1xf64>
// CHECK: hlfir.destroy
func.func @self_reference(%dest: !fir.ref<!fir.array<1xf64>>) {
  %one = arith.constant 1 : index
  %must_free = arith.constant true
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<1xf64>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1xf64>>, !fir.heap<!fir.array<1xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  // Reading the destination during construction prevents buffer reuse.
  %source = hlfir.designate %dest (%one) : (!fir.ref<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  %value = fir.load %source : !fir.ref<f64>
  hlfir.assign %value to %element : f64, !fir.ref<f64>
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.heap<!fir.array<1xf64>>, i1) -> !hlfir.expr<1xf64>
  hlfir.assign %expr to %dest : !hlfir.expr<1xf64>, !fir.ref<!fir.array<1xf64>>
  hlfir.destroy %expr : !hlfir.expr<1xf64>
  return
}

// Keep the temporary: escaped use of temporary.
// CHECK-LABEL: func.func @escaped_buffer(
// CHECK: fir.allocmem
// CHECK: hlfir.as_expr
// CHECK: hlfir.assign {{.*}} : !hlfir.expr<1xf64>
// CHECK: hlfir.destroy
func.func @escaped_buffer(%dest: !fir.ref<!fir.array<1xf64>>) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %must_free = arith.constant true
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<1xf64>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1xf64>>, !fir.heap<!fir.array<1xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  fir.call @escape(%tmp#1) : (!fir.heap<!fir.array<1xf64>>) -> ()
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.heap<!fir.array<1xf64>>, i1) -> !hlfir.expr<1xf64>
  hlfir.assign %expr to %dest : !hlfir.expr<1xf64>, !fir.ref<!fir.array<1xf64>>
  hlfir.destroy %expr : !hlfir.expr<1xf64>
  return
}

// Keep the temporary: use after assign.
// CHECK-LABEL: func.func @use_after_assign(
// CHECK: fir.allocmem
// CHECK: hlfir.as_expr
// CHECK: hlfir.assign {{.*}} : !hlfir.expr<1xf64>
// CHECK: hlfir.destroy
func.func @use_after_assign(%dest: !fir.ref<!fir.array<1xf64>>) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %must_free = arith.constant true
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<1xf64>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1xf64>>, !fir.heap<!fir.array<1xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.heap<!fir.array<1xf64>>, i1) -> !hlfir.expr<1xf64>
  hlfir.assign %expr to %dest : !hlfir.expr<1xf64>, !fir.ref<!fir.array<1xf64>>
  %late = fir.load %element : !fir.ref<f64>
  fir.call @consume(%late) : (f64) -> ()
  hlfir.destroy %expr : !hlfir.expr<1xf64>
  return
}

// Keep the temporary: %expr is used outside of assign.
// CHECK-LABEL: func.func @extra_expression_user(
// CHECK: fir.allocmem
// CHECK: hlfir.as_expr
// CHECK: hlfir.assign {{.*}} : !hlfir.expr<1xf64>
// CHECK: hlfir.destroy
func.func @extra_expression_user(%dest: !fir.ref<!fir.array<1xf64>>) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %must_free = arith.constant true
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<1xf64>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1xf64>>, !fir.heap<!fir.array<1xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.heap<!fir.array<1xf64>>, i1) -> !hlfir.expr<1xf64>
  hlfir.assign %expr to %dest : !hlfir.expr<1xf64>, !fir.ref<!fir.array<1xf64>>
  fir.call @use_expr(%expr) : (!hlfir.expr<1xf64>) -> ()
  hlfir.destroy %expr : !hlfir.expr<1xf64>
  return
}

// Keep the temporary: not a move as_expr.
// CHECK-LABEL: func.func @not_moved(
// CHECK: fir.allocmem
// CHECK: hlfir.as_expr
// CHECK: hlfir.assign {{.*}} : !hlfir.expr<1xf64>
// CHECK: hlfir.destroy
func.func @not_moved(%dest: !fir.ref<!fir.array<1xf64>>) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<1xf64>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1xf64>>, !fir.heap<!fir.array<1xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  %expr = hlfir.as_expr %tmp#0 : (!fir.heap<!fir.array<1xf64>>) -> !hlfir.expr<1xf64>
  hlfir.assign %expr to %dest : !hlfir.expr<1xf64>, !fir.ref<!fir.array<1xf64>>
  hlfir.destroy %expr : !hlfir.expr<1xf64>
  return
}

// Keep the temporary: explicit free.
// CHECK-LABEL: func.func @explicit_free(
// CHECK: fir.allocmem
// CHECK: hlfir.as_expr
// CHECK: hlfir.assign {{.*}} : !hlfir.expr<1xf64>
// CHECK: hlfir.destroy
func.func @explicit_free(%dest: !fir.ref<!fir.array<1xf64>>) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %must_free = arith.constant false
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<1xf64>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1xf64>>, !fir.heap<!fir.array<1xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.heap<!fir.array<1xf64>>, i1) -> !hlfir.expr<1xf64>
  hlfir.assign %expr to %dest : !hlfir.expr<1xf64>, !fir.ref<!fir.array<1xf64>>
  hlfir.destroy %expr : !hlfir.expr<1xf64>
  // The allocation is freed explicitly instead of by hlfir.destroy.
  fir.freemem %storage : !fir.heap<!fir.array<1xf64>>
  return
}

// Keep the temporary: different block.
// CHECK-LABEL: func.func @different_block(
// CHECK: fir.allocmem
// CHECK: hlfir.as_expr
// CHECK: hlfir.assign {{.*}} : !hlfir.expr<1xf64>
// CHECK: hlfir.destroy
func.func @different_block(%dest: !fir.ref<!fir.array<1xf64>>) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %must_free = arith.constant true
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<1xf64>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1xf64>>, !fir.heap<!fir.array<1xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.heap<!fir.array<1xf64>>, i1) -> !hlfir.expr<1xf64>
  cf.br ^assign
^assign:
  hlfir.assign %expr to %dest : !hlfir.expr<1xf64>, !fir.ref<!fir.array<1xf64>>
  hlfir.destroy %expr : !hlfir.expr<1xf64>
  return
}

func.func private @escape(!fir.heap<!fir.array<1xf64>>)
func.func private @consume(f64)
func.func private @use_expr(!hlfir.expr<1xf64>)
