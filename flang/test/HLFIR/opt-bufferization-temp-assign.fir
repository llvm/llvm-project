// RUN: fir-opt --opt-bufferization %s | FileCheck %s

// Rebase a destination with non-default lower bounds before redirecting stores.
// CHECK-LABEL: func.func @non_default_lower_bounds(
// CHECK: %[[DEST:.*]]:2 = hlfir.declare
// CHECK-NOT: fir.allocmem
// CHECK: %[[SHIFT:.*]] = fir.shift %c1
// CHECK: %[[REBOX:.*]] = fir.rebox %[[DEST]]#0(%[[SHIFT]])
// CHECK: %[[ELEMENT:.*]] = hlfir.designate %[[REBOX]] (%c1)
// CHECK: hlfir.assign %{{.*}} to %[[ELEMENT]] : f32, !fir.ref<f32>
// CHECK-NOT: hlfir.as_expr
// CHECK-NOT: hlfir.assign
// CHECK-NOT: hlfir.destroy
// CHECK: return
func.func @non_default_lower_bounds(%dest: !fir.ref<!fir.array<1xf32>>, %value: f32) {
  %one = arith.constant 1 : index
  %lb = arith.constant -1 : index
  %true = arith.constant true
  %dest_shape = fir.shape_shift %lb, %one : (index, index) -> !fir.shapeshift<1>
  %dest_decl:2 = hlfir.declare %dest(%dest_shape) uniq_name("dest") : (!fir.ref<!fir.array<1xf32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<1xf32>>, !fir.ref<!fir.array<1xf32>>)
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<1xf32>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1xf32>>, !fir.heap<!fir.array<1xf32>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1xf32>>, index) -> !fir.ref<f32>
  hlfir.assign %value to %element : f32, !fir.ref<f32>
  %expr = hlfir.as_expr %tmp#0 move %true : (!fir.heap<!fir.array<1xf32>>, i1) -> !hlfir.expr<1xf32>
  hlfir.assign %expr to %dest_decl#0 : !hlfir.expr<1xf32>, !fir.box<!fir.array<1xf32>>
  hlfir.destroy %expr : !hlfir.expr<1xf32>
  return
}

// Scalar logical conversion should have its temporary removed.
// CHECK-LABEL: func.func @logical_constructor(
// CHECK-NOT: fir.allocmem
// CHECK: %[[BOX:.*]] = fir.embox %arg0
// CHECK: %[[REBOX:.*]] = fir.rebox %[[BOX]]
// CHECK: %[[ELEMENT:.*]] = hlfir.designate %[[REBOX]]
// CHECK: hlfir.assign %arg1 to %[[ELEMENT]] : i1, !fir.ref<!fir.logical<4>>
// CHECK-NOT: hlfir.as_expr
// CHECK-NOT: hlfir.assign
// CHECK-NOT: hlfir.destroy
// CHECK: return
func.func @logical_constructor(%dest: !fir.ref<!fir.array<1x!fir.logical<4>>>, %value: i1) {
  %one = arith.constant 1 : index
  %true = arith.constant true
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<1x!fir.logical<4>>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1x!fir.logical<4>>>, !fir.heap<!fir.array<1x!fir.logical<4>>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
  hlfir.assign %value to %element : i1, !fir.ref<!fir.logical<4>>
  %expr = hlfir.as_expr %tmp#0 move %true : (!fir.heap<!fir.array<1x!fir.logical<4>>>, i1) -> !hlfir.expr<1x!fir.logical<4>>
  hlfir.assign %expr to %dest : !hlfir.expr<1x!fir.logical<4>>, !fir.ref<!fir.array<1x!fir.logical<4>>>
  hlfir.destroy %expr : !hlfir.expr<1x!fir.logical<4>>
  return
}

// Hoist a section of a non-default-bound array and redirect nested writes.
// CHECK-LABEL: func.func @section_constructor_custom_lb(
// CHECK: %[[DEST:.*]]:2 = hlfir.declare
// CHECK-NOT: fir.allocmem
// CHECK: %[[SECTION:.*]] = hlfir.designate %[[DEST]]#0
// CHECK: %[[BOX:.*]] = fir.embox %[[SECTION]]
// CHECK: %[[SHIFT:.*]] = fir.shift %c1
// CHECK: %[[REBOX:.*]] = fir.rebox %[[BOX]](%[[SHIFT]])
// CHECK: fir.do_loop %[[I:.*]] =
// CHECK: %[[ELEMENT:.*]] = hlfir.designate %[[REBOX]] (%[[I]])
// CHECK: hlfir.assign %arg1 to %[[ELEMENT]] : f32, !fir.ref<f32>
// CHECK-NOT: hlfir.as_expr
// CHECK-NOT: hlfir.assign
// CHECK-NOT: hlfir.destroy
// CHECK: return
func.func @section_constructor_custom_lb(%dest: !fir.ref<!fir.array<8xf32>>, %value: f32) {
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %eight = arith.constant 8 : index
  %lb = arith.constant -2 : index
  %true = arith.constant true
  %dest_shape = fir.shape_shift %lb, %eight : (index, index) -> !fir.shapeshift<1>
  %dest_decl:2 = hlfir.declare %dest(%dest_shape) uniq_name("dest") : (!fir.ref<!fir.array<8xf32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<8xf32>>, !fir.ref<!fir.array<8xf32>>)
  %shape = fir.shape %four : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<4xf32>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<4xf32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<4xf32>>, !fir.heap<!fir.array<4xf32>>)
  fir.do_loop %i = %one to %four step %one {
    %element = hlfir.designate %tmp#0 (%i) : (!fir.heap<!fir.array<4xf32>>, index) -> !fir.ref<f32>
    hlfir.assign %value to %element : f32, !fir.ref<f32>
  }
  %expr = hlfir.as_expr %tmp#0 move %true : (!fir.heap<!fir.array<4xf32>>, i1) -> !hlfir.expr<4xf32>
  %section = hlfir.designate %dest_decl#0 (%one:%four:%one) shape %shape : (!fir.box<!fir.array<8xf32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<4xf32>>
  hlfir.assign %expr to %section : !hlfir.expr<4xf32>, !fir.ref<!fir.array<4xf32>>
  hlfir.destroy %expr : !hlfir.expr<4xf32>
  return
}

// Freeing a distinct allocation between construction and assignment must not
// prevent removal of the temporary. Use the same array type so rejection comes from the
// allocation, not a difference in element types.
// CHECK-LABEL: func.func @unrelated_allocation_freed(
// CHECK-SAME: %[[DEST:[^:]+]]: !fir.ref<!fir.array<1xf64>>
// CHECK-NOT: fir.allocmem
// CHECK: %[[OTHER:.*]] = fir.allocmem !fir.array<1xf64>
// CHECK-NEXT: %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
// CHECK-NEXT: %[[EMBOX:.*]] = fir.embox %[[DEST]](%[[SHAPE]]) : (!fir.ref<!fir.array<1xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<1xf64>>
// CHECK-NEXT: %[[SHIFT:.*]] = fir.shift %c1 : (index) -> !fir.shift<1>
// CHECK-NEXT: %[[REBOX:.*]] = fir.rebox %[[EMBOX]](%[[SHIFT]]) : (!fir.box<!fir.array<1xf64>>, !fir.shift<1>) -> !fir.box<!fir.array<1xf64>>
// CHECK-NEXT: %[[ELEMENT:.*]] = hlfir.designate %[[REBOX]] (%{{.*}}) : (!fir.box<!fir.array<1xf64>>, index) -> !fir.ref<f64>
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

// Minimal eligible construction shared by some of the cases below.
// CHECK-LABEL: func.func @simple_constructor(
// CHECK-SAME: %[[DEST:[^:]+]]: !fir.ref<!fir.array<1xf64>>
// CHECK-NOT: fir.allocmem
// CHECK-NOT: fir.convert
// CHECK-NOT: hlfir.declare
// CHECK: %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
// CHECK: %[[EMBOX:.*]] = fir.embox %[[DEST]](%[[SHAPE]]) : (!fir.ref<!fir.array<1xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<1xf64>>
// CHECK: %[[SHIFT:.*]] = fir.shift %c1 : (index) -> !fir.shift<1>
// CHECK: %[[REBOX:.*]] = fir.rebox %[[EMBOX]](%[[SHIFT]]) : (!fir.box<!fir.array<1xf64>>, !fir.shift<1>) -> !fir.box<!fir.array<1xf64>>
// CHECK: %[[ELEMENT:.*]] = hlfir.designate %[[REBOX]] (%{{.*}}) : (!fir.box<!fir.array<1xf64>>, index) -> !fir.ref<f64>
// CHECK: hlfir.assign {{.*}} to %[[ELEMENT]] : f64, !fir.ref<f64>
// CHECK-NOT: hlfir.as_expr
// CHECK-NOT: hlfir.assign
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

// Keep the temporary when a destination section bound is computed after the
// allocation.
// CHECK-LABEL: func.func @section_constructor_late_bounds(
// CHECK: %[[DEST:.*]]:2 = hlfir.declare
// CHECK: %[[STORAGE:.*]] = fir.allocmem !fir.array<4xf32>
// CHECK: %[[TMP:.*]]:2 = hlfir.declare %[[STORAGE]]
// CHECK: %[[EXPR:.*]] = hlfir.as_expr %[[TMP]]#0 move
// CHECK: %[[START:.*]] = arith.addi
// CHECK: %[[END:.*]] = arith.addi %[[START]],
// CHECK: %[[SECTION:.*]] = hlfir.designate %[[DEST]]#0 (%[[START]]:%[[END]]:%{{.*}})
// CHECK: hlfir.assign %[[EXPR]] to %[[SECTION]] : !hlfir.expr<4xf32>, !fir.ref<!fir.array<4xf32>>
// CHECK: hlfir.destroy %[[EXPR]] : !hlfir.expr<4xf32>
// CHECK: return
func.func @section_constructor_late_bounds(%dest: !fir.ref<!fir.array<8xf32>>, %offset: index) {
  %one = arith.constant 1 : index
  %three = arith.constant 3 : index
  %four = arith.constant 4 : index
  %eight = arith.constant 8 : index
  %zero = arith.constant 0.0 : f32
  %must_free = arith.constant true
  %dest_shape = fir.shape %eight : (index) -> !fir.shape<1>
  %dest_decl:2 = hlfir.declare %dest(%dest_shape) uniq_name("dest") : (!fir.ref<!fir.array<8xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<8xf32>>, !fir.ref<!fir.array<8xf32>>)
  %shape = fir.shape %four : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<4xf32>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<4xf32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<4xf32>>, !fir.heap<!fir.array<4xf32>>)
  fir.do_loop %i = %one to %four step %one {
    %element = hlfir.designate %tmp#0 (%i) : (!fir.heap<!fir.array<4xf32>>, index) -> !fir.ref<f32>
    hlfir.assign %zero to %element : f32, !fir.ref<f32>
  }
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.heap<!fir.array<4xf32>>, i1) -> !hlfir.expr<4xf32>
  %start = arith.addi %offset, %one : index
  %end = arith.addi %start, %three : index
  %section = hlfir.designate %dest_decl#0 (%start:%end:%one) shape %shape : (!fir.ref<!fir.array<8xf32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<4xf32>>
  hlfir.assign %expr to %section : !hlfir.expr<4xf32>, !fir.ref<!fir.array<4xf32>>
  hlfir.destroy %expr : !hlfir.expr<4xf32>
  return
}


// Keep the temporary when its lower bound is not one. Replacing it with the
// one-based destination view would leave the construction subscript with
// lowerbound -1 unchanged.
// CHECK-LABEL: func.func @non_default_temporary_lower_bounds(
// CHECK-SAME: %[[DEST:[^:]+]]: !fir.ref<!fir.array<1xf64>>
// CHECK: %[[LB:.*]] = arith.constant -1 : index
// CHECK: %[[SHAPE:.*]] = fir.shape_shift %[[LB]], %{{.*}}
// CHECK: %[[STORAGE:.*]] = fir.allocmem !fir.array<1xf64>
// CHECK: %[[TMP:.*]]:2 = hlfir.declare %[[STORAGE]](%[[SHAPE]])
// CHECK: %[[ELEMENT:.*]] = hlfir.designate %[[TMP]]#0 (%[[LB]])
// CHECK: hlfir.assign {{.*}} to %[[ELEMENT]] : f64, !fir.ref<f64>
// CHECK: %[[EXPR:.*]] = hlfir.as_expr %[[TMP]]#0 move
// CHECK: hlfir.assign %[[EXPR]] to %[[DEST]] : !hlfir.expr<1xf64>, !fir.ref<!fir.array<1xf64>>
// CHECK: hlfir.destroy %[[EXPR]] : !hlfir.expr<1xf64>
// CHECK: return
func.func @non_default_temporary_lower_bounds(%dest: !fir.ref<!fir.array<1xf64>>) {
  %one = arith.constant 1 : index
  %minus_one = arith.constant -1 : index
  %zero = arith.constant 0.0 : f64
  %must_free = arith.constant true
  %shape = fir.shape_shift %minus_one, %one : (index, index) -> !fir.shapeshift<1>
  %storage = fir.allocmem !fir.array<1xf64>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf64>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<1xf64>>, !fir.heap<!fir.array<1xf64>>)
  %element = hlfir.designate %tmp#0 (%minus_one) : (!fir.box<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.box<!fir.array<1xf64>>, i1) -> !hlfir.expr<1xf64>
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

// Keep the temporary: a call with unknown effects, even without a temporary use.
// CHECK-LABEL: func.func @unknown_effects_call(
// CHECK: fir.allocmem
// CHECK: fir.call @consume({{.*}}) : (f64) -> ()
// CHECK: hlfir.as_expr
// CHECK: hlfir.assign {{.*}} : !hlfir.expr<1xf64>
// CHECK: hlfir.destroy
func.func @unknown_effects_call(%dest: !fir.ref<!fir.array<1xf64>>) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %must_free = arith.constant true
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<1xf64>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1xf64>>, !fir.heap<!fir.array<1xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  fir.call @consume(%zero) : (f64) -> ()
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.heap<!fir.array<1xf64>>, i1) -> !hlfir.expr<1xf64>
  hlfir.assign %expr to %dest : !hlfir.expr<1xf64>, !fir.ref<!fir.array<1xf64>>
  hlfir.destroy %expr : !hlfir.expr<1xf64>
  return
}

// Keep the temporary: the destination has a different element type.
// CHECK-LABEL: func.func @different_destination_element_type(
// CHECK: fir.allocmem !fir.array<1xf64>
// CHECK: hlfir.as_expr
// CHECK: hlfir.assign {{.*}} : !hlfir.expr<1xf64>, !fir.ref<!fir.array<1xf32>>
// CHECK: hlfir.destroy
func.func @different_destination_element_type(%dest: !fir.ref<!fir.array<1xf32>>) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %must_free = arith.constant true
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<1xf64>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1xf64>>, !fir.heap<!fir.array<1xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.heap<!fir.array<1xf64>>, i1) -> !hlfir.expr<1xf64>
  hlfir.assign %expr to %dest : !hlfir.expr<1xf64>, !fir.ref<!fir.array<1xf32>>
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

// Keep the temporary: temp and dest have known ranks but unknown extents at
// compile time.
// CHECK-LABEL: func.func @unknown_extent_dest_and_tmp(
// CHECK: fir.allocmem
// CHECK: hlfir.as_expr
// CHECK: hlfir.assign {{.*}} : !hlfir.expr<?xf64>, !fir.box<!fir.array<?xf64>>
// CHECK: hlfir.destroy
// CHECK: return
func.func @unknown_extent_dest_and_tmp(%dest: !fir.box<!fir.array<?xf64>>, %n: index) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %must_free = arith.constant true
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<?xf64>, %n
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.heap<!fir.array<?xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.box<!fir.array<?xf64>>, i1) -> !hlfir.expr<?xf64>
  hlfir.assign %expr to %dest : !hlfir.expr<?xf64>, !fir.box<!fir.array<?xf64>>
  hlfir.destroy %expr : !hlfir.expr<?xf64>
  return
}

// Keep the temporary: it has known rank and known extent
// but the destination has unknown extent.
// CHECK-LABEL: func.func @unknown_extent_dest_known_tmp(
// CHECK: hlfir.as_expr
// CHECK: hlfir.assign {{.*}} : !hlfir.expr<1xf64>, !fir.box<!fir.array<?xf64>>
// CHECK: hlfir.destroy %{{.*}} : !hlfir.expr<1xf64>
// CHECK: return
func.func @unknown_extent_dest_known_tmp(%dest: !fir.box<!fir.array<?xf64>>, %n: index) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %must_free = arith.constant true
  %shape = fir.shape %one : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<1xf64>
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<1xf64>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1xf64>>, !fir.heap<!fir.array<1xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.heap<!fir.array<1xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.heap<!fir.array<1xf64>>, i1) -> !hlfir.expr<1xf64>
  hlfir.assign %expr to %dest : !hlfir.expr<1xf64>, !fir.box<!fir.array<?xf64>>
  hlfir.destroy %expr : !hlfir.expr<1xf64>
  return
}

// Keep the temporary: it has known rank and unknown extent.
// CHECK-LABEL: func.func @unknown_extent_tmp_known_dest(
// CHECK: fir.allocmem
// CHECK: hlfir.as_expr
// CHECK: hlfir.assign {{.*}} : !hlfir.expr<?xf64>, !fir.heap<!fir.array<1xf64>>
// CHECK: hlfir.destroy
// CHECK: return
func.func @unknown_extent_tmp_known_dest(%dest: !fir.heap<!fir.array<1xf64>>, %n: index) {
  %one = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %must_free = arith.constant true
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  %storage = fir.allocmem !fir.array<?xf64>, %n
  %tmp:2 = hlfir.declare %storage(%shape) uniq_name("tmp") : (!fir.heap<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.heap<!fir.array<?xf64>>)
  %element = hlfir.designate %tmp#0 (%one) : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
  hlfir.assign %zero to %element : f64, !fir.ref<f64>
  %expr = hlfir.as_expr %tmp#0 move %must_free : (!fir.box<!fir.array<?xf64>>, i1) -> !hlfir.expr<?xf64>
  hlfir.assign %expr to %dest : !hlfir.expr<?xf64>, !fir.heap<!fir.array<1xf64>>
  hlfir.destroy %expr : !hlfir.expr<?xf64>
  return
}

func.func private @escape(!fir.heap<!fir.array<1xf64>>)
func.func private @consume(f64)
func.func private @use_expr(!hlfir.expr<1xf64>)
