// Test scheduling of WHERE assignments where the LHS and RHS subscript
// indices are loads of the same scalar variable that is declared outside
// the WHERE construct. The two loads have different SSA values (CSE cannot
// merge them across the LHS and RHS regions), but the scheduling analysis
// recognizes them as equivalent values via the ArraySectionAnalyzer
// ValueEquivalenceCallback, so the conflict between the LHS and RHS array
// sections is "aligned" and no temporary needs to be saved for the RHS.
//
// RUN: fir-opt %s --lower-hlfir-ordered-assignments \
// RUN:   --debug-only=flang-ordered-assignment \
// RUN:   -flang-dbg-order-assignment-schedule-only 2>&1 | FileCheck %s
//
// REQUIRES: asserts

// Positive case: same scalar `i` loaded in the LHS and RHS yield regions of
// the same hlfir.region_assign. The hlfir.declare for `i` is outside the
// hlfir.where, so the analysis can prove that the two fir.load operations
// produce the same value.
func.func @test_same_index(%arg_var: !fir.box<!fir.array<?x?xf32>>,
                           %arg_i: !fir.ref<i32>,
                           %arg_mask: !fir.box<!fir.array<?x!fir.logical<4>>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dscope = fir.dummy_scope : !fir.dscope
  %var:2 = hlfir.declare %arg_var dummy_scope %dscope {uniq_name = "_QFtest_same_indexEvar"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  %i:2 = hlfir.declare %arg_i dummy_scope %dscope {uniq_name = "_QFtest_same_indexEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %mask:2 = hlfir.declare %arg_mask dummy_scope %dscope {uniq_name = "_QFtest_same_indexEmask"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  hlfir.where {
    hlfir.yield %mask#0 : !fir.box<!fir.array<?x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      %dims:3 = fir.box_dims %var#1, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
      %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
      %i_val = fir.load %i#0 : !fir.ref<i32>
      %i_idx = fir.convert %i_val : (i32) -> i64
      %slice = hlfir.designate %var#0 (%c1:%dims#1:%c1, %i_idx) shape %shape : (!fir.box<!fir.array<?x?xf32>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
      hlfir.yield %slice : !fir.box<!fir.array<?xf32>>
    } to {
      %dims:3 = fir.box_dims %var#1, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
      %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
      %i_val = fir.load %i#0 : !fir.ref<i32>
      %i_idx = fir.convert %i_val : (i32) -> i64
      %slice = hlfir.designate %var#0 (%c1:%dims#1:%c1, %i_idx) shape %shape : (!fir.box<!fir.array<?x?xf32>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
      hlfir.yield %slice : !fir.box<!fir.array<?xf32>>
    }
  }
  return
}

// CHECK-LABEL: scheduling where in test_same_index ------------
// CHECK: conflict (aligned)
// CHECK: run 1 evaluate: where/region_assign1
// CHECK-NOT: save
// CHECK-NOT: run {{[2-9]}}

// Positive case with an arith expression on the index. The recursion through
// mlir::OperationEquivalence::isEquivalentTo matches the two arith.addi ops
// (one in the LHS region, one in the RHS region), then their %i_val operands
// via the ValueEquivalenceCallback's load-equivalence rule.
func.func @test_same_index_expr(%arg_var: !fir.box<!fir.array<?x?xf32>>,
                                %arg_i: !fir.ref<i32>,
                                %arg_mask: !fir.box<!fir.array<?x!fir.logical<4>>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %dscope = fir.dummy_scope : !fir.dscope
  %var:2 = hlfir.declare %arg_var dummy_scope %dscope {uniq_name = "_QFtest_same_index_exprEvar"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  %i:2 = hlfir.declare %arg_i dummy_scope %dscope {uniq_name = "_QFtest_same_index_exprEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %mask:2 = hlfir.declare %arg_mask dummy_scope %dscope {uniq_name = "_QFtest_same_index_exprEmask"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  hlfir.where {
    hlfir.yield %mask#0 : !fir.box<!fir.array<?x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      %dims:3 = fir.box_dims %var#1, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
      %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
      %i_val = fir.load %i#0 : !fir.ref<i32>
      %i_plus = arith.addi %i_val, %c1_i32 : i32
      %i_idx = fir.convert %i_plus : (i32) -> i64
      %slice = hlfir.designate %var#0 (%c1:%dims#1:%c1, %i_idx) shape %shape : (!fir.box<!fir.array<?x?xf32>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
      hlfir.yield %slice : !fir.box<!fir.array<?xf32>>
    } to {
      %dims:3 = fir.box_dims %var#1, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
      %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
      %i_val = fir.load %i#0 : !fir.ref<i32>
      %i_plus = arith.addi %i_val, %c1_i32 : i32
      %i_idx = fir.convert %i_plus : (i32) -> i64
      %slice = hlfir.designate %var#0 (%c1:%dims#1:%c1, %i_idx) shape %shape : (!fir.box<!fir.array<?x?xf32>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
      hlfir.yield %slice : !fir.box<!fir.array<?xf32>>
    }
  }
  return
}

// CHECK-LABEL: scheduling where in test_same_index_expr ------------
// CHECK: conflict (aligned)
// CHECK: run 1 evaluate: where/region_assign1
// CHECK-NOT: save
// CHECK-NOT: run {{[2-9]}}

// Negative case: the triplets actually differ between the LHS and the RHS
// (the RHS reads from a shifted lower bound). The ValueEquivalenceCallback
// can match the loads of `i`, but the section triplets are not identical
// (different `lb`), so the analysis still has to fall back to "Unknown"
// overlap, which forces saving the RHS to a temporary. This confirms the
// callback only relaxes equality, never strengthens it.
func.func @test_disjoint_section(%arg_var: !fir.box<!fir.array<?x?xf32>>,
                                 %arg_i: !fir.ref<i32>,
                                 %arg_mask: !fir.box<!fir.array<?x!fir.logical<4>>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dscope = fir.dummy_scope : !fir.dscope
  %var:2 = hlfir.declare %arg_var dummy_scope %dscope {uniq_name = "_QFtest_disjoint_sectionEvar"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  %i:2 = hlfir.declare %arg_i dummy_scope %dscope {uniq_name = "_QFtest_disjoint_sectionEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %mask:2 = hlfir.declare %arg_mask dummy_scope %dscope {uniq_name = "_QFtest_disjoint_sectionEmask"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  hlfir.where {
    hlfir.yield %mask#0 : !fir.box<!fir.array<?x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      %dims:3 = fir.box_dims %var#1, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
      %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
      %i_val = fir.load %i#0 : !fir.ref<i32>
      %i_idx = fir.convert %i_val : (i32) -> i64
      // RHS section: var(2:dims, i) - lower bound differs from LHS.
      %slice = hlfir.designate %var#0 (%c2:%dims#1:%c1, %i_idx) shape %shape : (!fir.box<!fir.array<?x?xf32>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
      hlfir.yield %slice : !fir.box<!fir.array<?xf32>>
    } to {
      %dims:3 = fir.box_dims %var#1, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
      %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
      %i_val = fir.load %i#0 : !fir.ref<i32>
      %i_idx = fir.convert %i_val : (i32) -> i64
      // LHS section: var(1:dims, i).
      %slice = hlfir.designate %var#0 (%c1:%dims#1:%c1, %i_idx) shape %shape : (!fir.box<!fir.array<?x?xf32>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
      hlfir.yield %slice : !fir.box<!fir.array<?xf32>>
    }
  }
  return
}

// CHECK-LABEL: scheduling where in test_disjoint_section ------------
// CHECK: conflicting arrays:
// CHECK: run {{[0-9]+}} save {{.*}}: where/region_assign1/rhs
// CHECK: run {{[0-9]+}} evaluate: where/region_assign1
