// Test the separate-allocatable-assign pass.
// It should transform hlfir.assign ... realloc into conditional reallocation
// followed by a non-realloc hlfir.assign.

// RUN: fir-opt --separate-allocatable-assign %s | FileCheck %s

// Test: allocatable array assignment with elemental RHS
func.func @test_expr_rhs(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %a:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
  %c:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEc"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)

  %a_box = fir.load %a#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %dims:3 = fir.box_dims %a_box, %c0 : (!fir.box<!fir.heap<!fir.array<?xf64>>>, index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>

  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?xf64> {
  ^bb0(%i: index):
    %lb_offset = arith.subi %dims#0, %c1 : index
    %idx = arith.addi %i, %lb_offset : index
    %a_elem = hlfir.designate %a_box (%idx) : (!fir.box<!fir.heap<!fir.array<?xf64>>>, index) -> !fir.ref<f64>
    %a_val = fir.load %a_elem : !fir.ref<f64>
    %cos_val = math.cos %a_val fastmath<contract> : f64
    hlfir.yield_element %cos_val : f64
  }

  hlfir.assign %elemental to %c#0 realloc : !hlfir.expr<?xf64>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  hlfir.destroy %elemental : !hlfir.expr<?xf64>
  return
}

// CHECK-LABEL: func.func @test_expr_rhs
// The realloc assign should be separated into realloc + non-realloc assign.
// CHECK-NOT: hlfir.assign{{.*}}realloc
// CHECK: fir.if
// CHECK: fir.allocmem
// CHECK: %[[BOX:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
// CHECK: hlfir.assign %{{.*}} to %[[BOX]] : !hlfir.expr<?xf64>, !fir.box<!fir.heap<!fir.array<?xf64>>>

// Test: allocatable array assignment with variable RHS
func.func @test_var_rhs(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, %arg1: !fir.ref<!fir.array<10xf64>>) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>

  %a:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
  %b:2 = hlfir.declare %arg1(%shape) {uniq_name = "_QFEb"} : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf64>>, !fir.ref<!fir.array<10xf64>>)

  hlfir.assign %b#0 to %a#0 realloc : !fir.ref<!fir.array<10xf64>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  return
}

// CHECK-LABEL: func.func @test_var_rhs
// CHECK-NOT: hlfir.assign{{.*}}realloc
// CHECK: fir.if
// CHECK: fir.allocmem
// CHECK: %[[BOX:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
// CHECK: hlfir.assign %{{.*}} to %[[BOX]] : !fir.ref<!fir.array<10xf64>>, !fir.box<!fir.heap<!fir.array<?xf64>>>

// Test: non-trivial element type should NOT be separated
func.func @test_nontrivial(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>

  %a:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>)

  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.char<1,10>> {
  ^bb0(%i: index):
    %str = fir.undefined !fir.char<1,10>
    hlfir.yield_element %str : !fir.char<1,10>
  }

  hlfir.assign %elemental to %a#0 realloc : !hlfir.expr<?x!fir.char<1,10>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  hlfir.destroy %elemental : !hlfir.expr<?x!fir.char<1,10>>
  return
}

// CHECK-LABEL: func.func @test_nontrivial
// Character types are not trivial, so the assign should remain
// CHECK: hlfir.assign %{{.*}} to %{{.*}} realloc

// Test: non-allocatable assign should NOT be modified
func.func @test_non_allocatable(%arg0: !fir.ref<!fir.array<10xf64>>, %arg1: !fir.ref<!fir.array<10xf64>>) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>

  %a:2 = hlfir.declare %arg0(%shape) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf64>>, !fir.ref<!fir.array<10xf64>>)
  %b:2 = hlfir.declare %arg1(%shape) {uniq_name = "_QFEb"} : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf64>>, !fir.ref<!fir.array<10xf64>>)

  hlfir.assign %b#0 to %a#0 : !fir.ref<!fir.array<10xf64>>, !fir.ref<!fir.array<10xf64>>
  return
}

// CHECK-LABEL: func.func @test_non_allocatable
// Non-allocatable assign should pass through unchanged
// CHECK: hlfir.assign %{{.*}} to %{{.*}} : !fir.ref<!fir.array<10xf64>>, !fir.ref<!fir.array<10xf64>>
// CHECK-NOT: fir.if

// Test: self-aliasing (a = a(:n)) should NOT be separated because realloc
// would free the old LHS storage that the RHS still references.
func.func @test_self_alias(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index

  %a:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %a_box = fir.load %a#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %dims:3 = fir.box_dims %a_box, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)

  %shape = fir.shape %c5 : (index) -> !fir.shape<1>
  %section = hlfir.designate %a_box (%c1:%c5:%c1) shape %shape : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>

  hlfir.assign %section to %a#0 realloc : !fir.box<!fir.array<5xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}

// CHECK-LABEL: func.func @test_self_alias
// Self-aliasing must keep the original realloc assign
// CHECK: hlfir.assign %{{.*}} to %{{.*}} realloc

// Test: expr RHS whose producing elemental reads from the LHS
// (e.g. a = a(:n) + 1).  The pass separates the reallocation regardless of
// any self-reference in the expression: keeping the (lazy) expression
// evaluation from being moved across the reallocation's deallocation is the
// responsibility of the hlfir.assign lowering / expression bufferization, not
// of this pass.  See the deallocation-conflict check in OptimizedBufferization.
func.func @test_self_alias_expr(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %one_i32 = arith.constant 1 : i32

  %a:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %a_box = fir.load %a#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %shape = fir.shape %c5 : (index) -> !fir.shape<1>

  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%i: index):
    %elem = hlfir.designate %a_box (%i) : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
    %val = fir.load %elem : !fir.ref<i32>
    %sum = arith.addi %val, %one_i32 : i32
    hlfir.yield_element %sum : i32
  }

  hlfir.assign %elemental to %a#0 realloc : !hlfir.expr<?xi32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  hlfir.destroy %elemental : !hlfir.expr<?xi32>
  return
}

// CHECK-LABEL: func.func @test_self_alias_expr
// Expression RHS is always separated; alias safety is left to bufferization.
// CHECK-NOT: hlfir.assign{{.*}}realloc
// CHECK: fir.if
// CHECK: fir.allocmem
// CHECK: %[[BOX:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: hlfir.assign %{{.*}} to %[[BOX]] : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>

// Test: lower bounds from RHS should be preserved during reallocation.
// source(10:12) has lower bound 10; dest should get lower bound 10 after
// dest = source.
func.func @test_lower_bounds(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.array<3xi32>>) {
  %c10 = arith.constant 10 : index
  %c3 = arith.constant 3 : index
  %shapeshift = fir.shape_shift %c10, %c3 : (index, index) -> !fir.shapeshift<1>

  %dest:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEdest"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %source:2 = hlfir.declare %arg1(%shapeshift) {uniq_name = "_QFEsource"} : (!fir.ref<!fir.array<3xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)

  hlfir.assign %source#0 to %dest#0 realloc : !fir.box<!fir.array<3xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}

// CHECK-LABEL: func.func @test_lower_bounds
// The realloc should be separated with lower bound 10 propagated.
// CHECK-NOT: hlfir.assign{{.*}}realloc
// CHECK: %[[C10:.*]] = arith.constant 10 : index
// CHECK: fir.if
// CHECK: fir.allocmem
// Lower bound 10 should appear in the embox/store of the new allocation.
// CHECK: %[[BOX:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: hlfir.assign %{{.*}} to %[[BOX]] : !fir.box<!fir.array<3xi32>>, !fir.box<!fir.heap<!fir.array<?xi32>>>

// Test: lower bounds from an hlfir.expr RHS that wraps a variable (via
// hlfir.as_expr) must be preserved during reallocation.  This is the
// structure-constructor case `t = x(f())` where `f` is an allocatable function
// result with a non-default lower bound: the allocatable component must take
// the source's bounds (F2018 7.5.10).  The pass must read the lower bound from
// the wrapped variable's descriptor instead of assuming 1.
func.func @test_expr_lower_bounds(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.array<3xi32>>) {
  %c10 = arith.constant 10 : index
  %c3 = arith.constant 3 : index
  %true = arith.constant true
  %shapeshift = fir.shape_shift %c10, %c3 : (index, index) -> !fir.shapeshift<1>

  %dest:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEdest"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %source:2 = hlfir.declare %arg1(%shapeshift) {uniq_name = "_QFEsource"} : (!fir.ref<!fir.array<3xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
  %expr = hlfir.as_expr %source#0 move %true : (!fir.box<!fir.array<3xi32>>, i1) -> !hlfir.expr<3xi32>

  hlfir.assign %expr to %dest#0 realloc temporary_lhs : !hlfir.expr<3xi32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  hlfir.destroy %expr : !hlfir.expr<3xi32>
  return
}

// CHECK-LABEL: func.func @test_expr_lower_bounds
// The realloc should be separated with the wrapped variable's lower bound 10.
// CHECK-NOT: hlfir.assign{{.*}}realloc
// CHECK: %[[C10:.*]] = arith.constant 10 : index
// CHECK: fir.if
// CHECK: fir.allocmem
// Lower bound 10 (not 1) must appear in the embox/store of the new allocation.
// CHECK: fir.shape_shift %[[C10]]
// CHECK: %[[BOX:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: hlfir.assign %{{.*}} to %[[BOX]] temporary_lhs : !hlfir.expr<3xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>
