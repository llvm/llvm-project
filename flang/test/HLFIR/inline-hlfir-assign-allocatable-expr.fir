// Test inlining of hlfir.assign for allocatable LHS with hlfir.expr RHS.
// This tests the -inline-hlfir-allocatable-expr-assign flag.

// RUN: fir-opt --inline-hlfir-assign %s | FileCheck %s --check-prefix=DEFAULT
// RUN: fir-opt -inline-hlfir-allocatable-expr-assign --inline-hlfir-assign %s | FileCheck %s --check-prefix=ENABLED

// Test case: c = cos(a) where c is allocatable
// This is derived from the flang-529628 test case.
// The hlfir.elemental produces an hlfir.expr which doesn't alias with
// memory, so the assignment can be safely inlined.

func.func @test_allocatable_elemental_assign(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Declare the allocatables
  %a:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
  %c:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEc"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)

  // Load a to get its shape
  %a_box = fir.load %a#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %dims:3 = fir.box_dims %a_box, %c0 : (!fir.box<!fir.heap<!fir.array<?xf64>>>, index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>

  // Create elemental: cos(a)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?xf64> {
  ^bb0(%i: index):
    %lb_offset = arith.subi %dims#0, %c1 : index
    %idx = arith.addi %i, %lb_offset : index
    %a_elem = hlfir.designate %a_box (%idx) : (!fir.box<!fir.heap<!fir.array<?xf64>>>, index) -> !fir.ref<f64>
    %a_val = fir.load %a_elem : !fir.ref<f64>
    %cos_val = math.cos %a_val fastmath<contract> : f64
    hlfir.yield_element %cos_val : f64
  }

  // Assign elemental result to allocatable c
  hlfir.assign %elemental to %c#0 realloc : !hlfir.expr<?xf64>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  hlfir.destroy %elemental : !hlfir.expr<?xf64>
  return
}

// DEFAULT-LABEL: func.func @test_allocatable_elemental_assign
// By default (without the option), the allocatable assign should NOT be inlined
// DEFAULT: hlfir.assign %{{.*}} to %{{.*}} realloc : !hlfir.expr<?xf64>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>

// ENABLED-LABEL: func.func @test_allocatable_elemental_assign
// With the option enabled, the assign should be inlined.
// The generated code uses genReallocIfNeeded which creates fir.if for
// allocation checking and a loop for assignment.
// ENABLED-NOT: hlfir.assign %{{.*}} realloc
// ENABLED: hlfir.elemental
// ENABLED: %[[ORIG_MEM:.*]] = fir.box_addr %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.heap<!fir.array<?xf64>>
// ENABLED: %[[IF_RET:.*]]:2 = fir.if %[[COND:.*]] -> (i1, !fir.heap<!fir.array<?xf64>>) {
// ENABLED: fir.do_loop
// ENABLED: hlfir.apply
// ENABLED: hlfir.designate
// ENABLED: hlfir.assign %{{.*}} : f64, !fir.ref<f64>
// ENABLED: fir.if %[[IF_RET]]#0 {
// ENABLED: fir.if %[[COND]] {
// ENABLED: fir.freemem %[[ORIG_MEM]] : !fir.heap<!fir.array<?xf64>>

// Test case: Non-trivial element type should NOT be inlined
func.func @test_allocatable_nontrivial_type(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>

  %a:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>)

  // Character elemental - should NOT be inlined even with the option
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.char<1,10>> {
  ^bb0(%i: index):
    %str = fir.undefined !fir.char<1,10>
    hlfir.yield_element %str : !fir.char<1,10>
  }

  hlfir.assign %elemental to %a#0 realloc : !hlfir.expr<?x!fir.char<1,10>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  hlfir.destroy %elemental : !hlfir.expr<?x!fir.char<1,10>>
  return
}

// Character types are not trivial, so this should never be inlined
// ENABLED-LABEL: func.func @test_allocatable_nontrivial_type
// ENABLED: hlfir.assign %{{.*}} to %{{.*}} realloc : !hlfir.expr<?x!fir.char<1,10>>


// Test case: Variable RHS (not hlfir.expr) should NOT be inlined by this pattern
func.func @test_allocatable_variable_rhs(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, %arg1: !fir.ref<!fir.array<10xf64>>) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>

  %a:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
  %b:2 = hlfir.declare %arg1(%shape) {uniq_name = "_QFEb"} : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf64>>, !fir.ref<!fir.array<10xf64>>)

  // Variable RHS - NOT an hlfir.expr, so the allocatable pattern should NOT match
  hlfir.assign %b#0 to %a#0 realloc : !fir.ref<!fir.array<10xf64>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  return
}

// Variable RHS should keep the original assign (allocatable pattern doesn't match)
// ENABLED-LABEL: func.func @test_allocatable_variable_rhs
// ENABLED: hlfir.assign %{{.*}} to %{{.*}} realloc : !fir.ref<!fir.array<10xf64>>
