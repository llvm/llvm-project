// RUN: fir-opt --inline-elementals %s | FileCheck %s

// Test inlining of hlfir.elemental into its hlfir.apply site when the 
// elemental has more than two users.

// Check successful inlining where hlfir.elemental survives because the
// 'associate' op is still using it.
func.func @test_inlining_use_mask(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10, %c10 : (index, index) -> !fir.shape<2>

  // Elemental Mask.
  %mask = hlfir.elemental %shape unordered : (!fir.shape<2>) -> !hlfir.expr<10x10x!fir.logical<4>> {
  ^bb0(%i: index, %j: index):
    %val = hlfir.designate %arg0 (%i, %j) : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
    %load = fir.load %val : !fir.ref<i32>
    %ref = fir.load %arg1 : !fir.ref<i32>
    %cmp = arith.cmpi eq, %load, %ref : i32
    %res = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %res : !fir.logical<4>
  }

  // Extra User - This keeps the elemental alive even after inlining the apply
  // site. Total uses = 3 (associate, apply, destroy).
  %extra:3 = hlfir.associate %mask : (!hlfir.expr<10x10x!fir.logical<4>>) -> (!fir.box<!fir.array<10x10x!fir.logical<4>>>, !fir.ref<!fir.array<10x10x!fir.logical<4>>>, i1)

  // Apply inside a loop
  fir.do_loop %arg2 = %c1 to %c10 step %c1 {
    fir.do_loop %arg3 = %c1 to %c10 step %c1 {
      %applied = hlfir.apply %mask, %arg3, %arg2 : (!hlfir.expr<10x10x!fir.logical<4>>, index, index) -> !fir.logical<4>
      %dummy_ref = fir.alloca !fir.logical<4>
      fir.store %applied to %dummy_ref : !fir.ref<!fir.logical<4>>
    }
 }

  hlfir.end_associate %extra#0, %extra#2 : !fir.box<!fir.array<10x10x!fir.logical<4>>>, i1
  hlfir.destroy %mask : !hlfir.expr<10x10x!fir.logical<4>>
  return
}
// CHECK-LABEL: func.func @test_inlining_use_mask
// CHECK: %[[MASK:.*]] = hlfir.elemental
// CHECK: hlfir.associate %[[MASK]]
// CHECK: fir.do_loop
// CHECK:   fir.do_loop
// CHECK-NOT: hlfir.apply
// CHECK:     %[[VAL:.*]] = hlfir.designate %arg0
// CHECK:     %[[LOAD:.*]] = fir.load %[[VAL]]
// CHECK:     %[[REF:.*]] = fir.load %arg1
// CHECK:     %[[CMP:.*]] = arith.cmpi eq, %[[LOAD]], %[[REF]]
// CHECK:     %[[RES:.*]] = fir.convert %[[CMP]]
// CHECK:     fir.store %[[RES]]
// CHECK-NOT: hlfir.destroy %[[MASK]]

// Test elemental removal as use_count becomes zero.
func.func @test_inlining_elemental_cleanup(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10, %c10 : (index, index) -> !fir.shape<2>

  // Elemental Mask.
  %mask = hlfir.elemental %shape unordered : (!fir.shape<2>) -> !hlfir.expr<10x10x!fir.logical<4>> {
  ^bb0(%i: index, %j: index):
    %val = hlfir.designate %arg0 (%i, %j) : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
    %load = fir.load %val : !fir.ref<i32>
    %ref = fir.load %arg1 : !fir.ref<i32>
    %cmp = arith.cmpi eq, %load, %ref : i32
    %res = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %res : !fir.logical<4>
  }

  %extra = fir.convert %mask : (!hlfir.expr<10x10x!fir.logical<4>>) -> !hlfir.expr<10x10x!fir.logical<4>>

  // Apply Site.
  fir.do_loop %arg2 = %c1 to %c10 step %c1 {
    fir.do_loop %arg3 = %c1 to %c10 step %c1 {
      %applied = hlfir.apply %mask, %arg3, %arg2 : (!hlfir.expr<10x10x!fir.logical<4>>, index, index) -> !fir.logical<4>
      %dummy_ref = fir.alloca !fir.logical<4>
      fir.store %applied to %dummy_ref : !fir.ref<!fir.logical<4>>
    }
  }

  hlfir.destroy %mask : !hlfir.expr<10x10x!fir.logical<4>>
  return
}
// CHECK-LABEL: func.func @test_inlining_elemental_cleanup
// CHECK-NOT:     hlfir.elemental
// CHECK-NOT:     fir.convert
// CHECK:         fir.do_loop
// CHECK:           fir.do_loop
// CHECK-NOT:         hlfir.apply
// CHECK:             arith.cmpi eq
// CHECK-NOT:     hlfir.destroy

// Check that inlining is blocked when there is more than one hlfir.apply
// site for the same elemental.
func.func @test_multi_apply_no_inlining(%arg0: !hlfir.expr<?xi32>, %target: i32, %shape: !fir.shape<1>) -> (i1, i1) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // Producer (Elemental)
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
    ^bb0(%i: index):
      %val = hlfir.apply %arg0, %i : (!hlfir.expr<?xi32>, index) -> i32
      %cmp = arith.cmpi eq, %val, %target : i32
      %log = fir.convert %cmp : (i1) -> !fir.logical<4>
      hlfir.yield_element %log : !fir.logical<4>
  }

  // First Apply Site.
  %apply1 = hlfir.apply %mask, %c1 : (!hlfir.expr<?x!fir.logical<4>>, index) -> !fir.logical<4>
  %cond1 = fir.convert %apply1 : (!fir.logical<4>) -> i1

  // Second Apply Site.
  %apply2 = hlfir.apply %mask, %c2 : (!hlfir.expr<?x!fir.logical<4>>, index) -> !fir.logical<4>
  %cond2 = fir.convert %apply2 : (!fir.logical<4>) -> i1

  // Destroy.
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>

  return %cond1, %cond2 : i1, i1
}
// CHECK-LABEL: func.func @test_multi_apply_no_inlining(
// CHECK-SAME:  %[[ARG0:.*]]: !hlfir.expr<?xi32>, %[[TARGET:.*]]: i32, %[[SHAPE:.*]]: !fir.shape<1>)
// CHECK:       %[[MASK:.*]] = hlfir.elemental %[[SHAPE]]
// CHECK:       %[[A1:.*]] = hlfir.apply %[[MASK]], %{{.*}}
// CHECK:       %[[A2:.*]] = hlfir.apply %[[MASK]], %{{.*}}
// CHECK:       hlfir.destroy %[[MASK]]

// Check inlining one elemental into another.
// a = b * c + d
// Declare a global symbol to store the intermediate mask.
fir.global @mask_storage : !hlfir.expr<10x10x!fir.logical<4>>
func.func @test_nested_elemental(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10, %c10 : (index, index) -> !fir.shape<2>

  // The Elemental Mask (b * c)
  %mask = hlfir.elemental %shape unordered : (!fir.shape<2>) -> !hlfir.expr<10x10x!fir.logical<4>> {
  ^bb0(%i: index, %j: index):
    %val = hlfir.designate %arg0 (%i, %j) : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
    %load = fir.load %val : !fir.ref<i32>
    %ref = fir.load %arg1 : !fir.ref<i32>
    %cmp = arith.cmpi eq, %load, %ref : i32
    %res = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %res : !fir.logical<4>
  }

  // Total users - 1. fir.store, 2. hlfir.apply, 3. hlfir.destroy.
  %ptr = fir.address_of(@mask_storage) : !fir.ref<!hlfir.expr<10x10x!fir.logical<4>>>
  fir.store %mask to %ptr : !fir.ref<!hlfir.expr<10x10x!fir.logical<4>>>

  // Target loop using the mask.
  fir.do_loop %arg2 = %c1 to %c10 step %c1 {
    fir.do_loop %arg3 = %c1 to %c10 step %c1 {
      // CHECK-NOT: hlfir.apply
      %applied = hlfir.apply %mask, %arg3, %arg2 : (!hlfir.expr<10x10x!fir.logical<4>>, index, index) -> !fir.logical<4>
      %dummy_ref = fir.alloca !fir.logical<4>
      fir.store %applied to %dummy_ref : !fir.ref<!fir.logical<4>>
    }
  }

  hlfir.destroy %mask : !hlfir.expr<10x10x!fir.logical<4>>
  return
}
// CHECK-LABEL: func.func @test_nested_elemental
// CHECK:         hlfir.elemental
// CHECK-NOT:       hlfir.apply
// CHECK:           arith.cmpi eq
// CHECK-NOT:         hlfir.destroy

// Inlining into a single hlfir.apply.
// a = (b * c)[1]
func.func @test_scalar_apply_inline(%b: !fir.box<!fir.array<?xf32>>, %c: !fir.box<!fir.array<?xf32>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>

  // Elemental (b * c)
  %prod = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ^bb0(%i: index):
    %b_ref = hlfir.designate %b (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %b_val = fir.load %b_ref : !fir.ref<f32>
    %c_ref = hlfir.designate %c (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %c_val = fir.load %c_ref : !fir.ref<f32>
    %mul = arith.mulf %b_val, %c_val : f32
    hlfir.yield_element %mul : f32
  }

  // Total users = 3 (store, apply, destroy).
  %ptr = fir.address_of(@scalar_storage) : !fir.ref<!hlfir.expr<10xf32>>
  fir.store %prod to %ptr : !fir.ref<!hlfir.expr<10xf32>>

  // Scalar apply site - a = (b * c)(1)
  // CHECK-NOT: hlfir.apply
  %scalar_val = hlfir.apply %prod, %c1 : (!hlfir.expr<10xf32>, index) -> f32
  
  // Use the scalar result.
  %dummy_ref = fir.alloca f32
  fir.store %scalar_val to %dummy_ref : !fir.ref<f32>

  hlfir.destroy %prod : !hlfir.expr<10xf32>
  return
}
// CHECK-LABEL: func.func @test_scalar_apply_inline
// CHECK:         %[[MASK:.*]] = hlfir.elemental
// CHECK:         fir.store %[[MASK]] to {{.*}}
// CHECK-NOT:     hlfir.apply
// CHECK:         arith.mulf
// CHECK-NOT:     hlfir.destroy

// Check long chains of elementals.
// subroutine reproducer(a)
//   real, dimension(:) :: a
//   a = sqrt(a * (a - 1))
// end subroutine
func.func @_QPreproducer(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"}) {
  %c0 = arith.constant 0 : index
  %f1 = arith.constant 1.0 : f32
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFreproducerEa"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %1:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  %2 = fir.shape %1#1 : (index) -> !fir.shape<1>

  // tmp1 = a - 1
  %tmp1 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%i: index):
    %a_ref = hlfir.designate %0#0 (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %a_val = fir.load %a_ref : !fir.ref<f32>
    %sub = arith.subf %a_val, %f1 : f32
    hlfir.yield_element %sub : f32
  }

  %dummy = hlfir.no_reassoc %tmp1 : !hlfir.expr<?xf32>

  // tmp2 = a * tmp1
  %tmp2 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%j: index):
    %a_ref_2 = hlfir.designate %0#0 (%j) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %t1_val = hlfir.apply %tmp1, %j : (!hlfir.expr<?xf32>, index) -> f32
    %a_val_2 = fir.load %a_ref_2 : !fir.ref<f32>
    %mul = arith.mulf %a_val_2, %t1_val : f32
    hlfir.yield_element %mul : f32
  }

  // tmp3 = sqrt(tmp2)
  %tmp3 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%k: index):
    %t2_val = hlfir.apply %tmp2, %k : (!hlfir.expr<?xf32>, index) -> f32
    %res = math.sqrt %t2_val : f32
    hlfir.yield_element %res : f32
  }

  // Final assignment.
  hlfir.assign %tmp3 to %0#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>

  hlfir.destroy %tmp3 : !hlfir.expr<?xf32>
  hlfir.destroy %tmp2 : !hlfir.expr<?xf32>
  hlfir.destroy %dummy: !hlfir.expr<?xf32>
  hlfir.destroy %tmp1 : !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL: func.func @_QPreproducer
// CHECK:         %[[TMP1:.*]] = hlfir.elemental
// CHECK:         hlfir.no_reassoc %[[TMP1]]
// CHECK-NOT:     hlfir.apply
// CHECK-DAG:     arith.subf
// CHECK-DAG:     arith.mulf
// CHECK-DAG:     math.sqrt
// CHECK:         hlfir.assign
// The apply site was inlined, so the elemental's lifecycle (destroy) 
// is removed even though metadata users like no_reassoc remain.
// CHECK-NOT:     hlfir.destroy %[[TMP1]]

// Check that the ordered elemental is not inlined into another:
// a = b + c + d (where b + c is ordered)
func.func private @persistent_user(!hlfir.expr<?xf32>)
func.func @test_noinline_ordered(%arg0: !hlfir.expr<?xf32>, %arg1: !hlfir.expr<?xf32>, %shape: !fir.shape<1>) {
  %c1 = arith.constant 1 : index

  // Producer (b + c) - ordered
  %el_a = hlfir.elemental %shape {ordered = true} : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
    ^bb0(%i: index):
      %0 = hlfir.apply %arg0, %i : (!hlfir.expr<?xf32>, index) -> f32
      %1 = hlfir.apply %arg1, %i : (!hlfir.expr<?xf32>, index) -> f32
      %sum = arith.addf %0, %1 : f32
      hlfir.yield_element %sum : f32
  }

  fir.call @persistent_user(%el_a) : (!hlfir.expr<?xf32>) -> ()

  // Consumer (el_a + d)
  %el_b = hlfir.elemental %shape : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
    ^bb0(%j: index):
      // This apply must remain because el_a is ordered.
      %a_val = hlfir.apply %el_a, %j : (!hlfir.expr<?xf32>, index) -> f32
      %total = arith.addf %a_val, %a_val : f32
      hlfir.yield_element %total : f32
  }

  hlfir.destroy %el_a : !hlfir.expr<?xf32>
  %val = hlfir.apply %el_b, %c1 : (!hlfir.expr<?xf32>, index) -> f32
  hlfir.destroy %el_b : !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL: func.func @test_noinline_ordered
// CHECK:         %[[PRODUCER:.*]] = hlfir.elemental %{{.*}} {ordered = true}
// CHECK:         fir.call @persistent_user(%[[PRODUCER]])
// CHECK:         %[[CONSUMER:.*]] = hlfir.elemental
// CHECK:           hlfir.apply %[[PRODUCER]], %{{.*}}
// CHECK:         hlfir.destroy %[[PRODUCER]]

// Check that the elemental is not inlined, because its array result
// must be finalized.
func.func @test_noinline_due_to_finalization(%arg0: !fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:f32}>>>, %shape: !fir.shape<1>) {
  %c1 = arith.constant 1 : index

  // Producer - Derived-type Elemental.
  %el = hlfir.elemental %shape : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<_QMtypesTt1{x:f32}>> {
  ^bb0(%i: index):
    %res = fir.alloca !fir.type<_QMtypesTt1{x:f32}>
    %ld = fir.load %res : !fir.ref<!fir.type<_QMtypesTt1{x:f32}>>
    hlfir.yield_element %ld : !fir.type<_QMtypesTt1{x:f32}>
  }

  fir.call @persistent_user(%el) : (!hlfir.expr<?x!fir.type<_QMtypesTt1{x:f32}>>) -> ()

  // This apply must remain because the elemental result requires finalization.
  %res_ptr = fir.alloca !fir.type<_QMtypesTt1{x:f32}>
  %apply = hlfir.apply %el, %c1 : (!hlfir.expr<?x!fir.type<_QMtypesTt1{x:f32}>>, index) -> !hlfir.expr<!fir.type<_QMtypesTt1{x:f32}>>
  hlfir.assign %apply to %res_ptr : !hlfir.expr<!fir.type<_QMtypesTt1{x:f32}>>, !fir.ref<!fir.type<_QMtypesTt1{x:f32}>>

  // Destroy with 'finalize' keyword, hlfir::elementalOpMustProduceTemp becomes
  // true.
  hlfir.destroy %el finalize : !hlfir.expr<?x!fir.type<_QMtypesTt1{x:f32}>>
  return
}
// CHECK-LABEL: func.func @test_noinline_due_to_finalization
// CHECK:         %[[EL:.*]] = hlfir.elemental
// CHECK:         fir.call @persistent_user(%[[EL]])
// CHECK:         %[[APPLY:.*]] = hlfir.apply %[[EL]], %{{.*}}
// CHECK:         hlfir.destroy %[[EL]] finalize
