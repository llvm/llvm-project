// RUN: fir-opt --inline-elementals %s | FileCheck %s

// Test inlining of hlfir.elemental into its hlfir.apply site when the
// elemental has more than two users.

// Test successful inlining with relaxed inlining.
func.func @test_safe_loop_inlining(%arg0: !fir.ref<!fir.array<10xf32>>, %i: index) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>

  // Elemental Mask.
  %elem = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
    ^bb0(%idx: index):
      %addr = fir.coordinate_of %arg0, %idx : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
      %val = fir.load %addr : !fir.ref<f32>
      hlfir.yield_element %val : f32
  }

  // User 1 - hlfir.apply inside nested loops (The target for inlining).
  fir.do_loop %arg1 = %c1 to %c10 step %c1 {
    fir.do_loop %arg2 = %c1 to %c10 step %c1 {
      %res = hlfir.apply %elem, %arg2 : (!hlfir.expr<10xf32>, index) -> f32
      %dummy = fir.alloca f32
      fir.store %res to %dummy : !fir.ref<f32>
    }
  }

  // User 2 - Associate (Simulating a shared mask use-case).
  %temp:3 = hlfir.associate %elem(%shape) : (!hlfir.expr<10xf32>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>, i1)

  // User 3 - Destroy
  hlfir.destroy %elem : !hlfir.expr<10xf32>

  hlfir.end_associate %temp#0, %temp#2 : !fir.ref<!fir.array<10xf32>>, i1
  return
}
// CHECK-LABEL: func.func @test_safe_loop_inlining
// CHECK: %[[ELEM:.*]] = hlfir.elemental
// CHECK: fir.do_loop
// CHECK:   fir.do_loop %[[INNER_IDX:.*]] =
// CHECK:     %[[ADDR:.*]] = fir.coordinate_of %arg0, %[[INNER_IDX]]
// CHECK:     %[[VAL:.*]] = fir.load %[[ADDR]]
// CHECK-NOT: hlfir.apply
// CHECK: hlfir.associate %[[ELEM]]
// CHECK-NOT: hlfir.destroy %[[ELEM]]

// Test blocking of incorrect inlining because of alias conflict.
func.func @test_unsafe_loop_alias_conflict(%arg0: !fir.ref<!fir.array<10xf32>>, %new_val: f32) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>

  // Elemental depends on the values in %arg0 (Producer).
  %elem = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
    ^bb0(%idx: index):
      %addr = fir.coordinate_of %arg0, %idx : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
      %val = fir.load %addr : !fir.ref<f32>
      hlfir.yield_element %val : f32
  }

  fir.do_loop %arg1 = %c1 to %c10 step %c1 {
    // We modify the array that the elemental needs to read from.
    // Inlining the elemental here would see the new value.
    %write_addr = fir.coordinate_of %arg0, %arg1 : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
    fir.store %new_val to %write_addr : !fir.ref<f32>

    // Target for inlining.
    %res = hlfir.apply %elem, %arg1 : (!hlfir.expr<10xf32>, index) -> f32

    %dummy = fir.alloca f32
    fir.store %res to %dummy : !fir.ref<f32>
  }

  hlfir.destroy %elem : !hlfir.expr<10xf32>
  return
}
// CHECK-LABEL: func.func @test_unsafe_loop_alias_conflict
// Elemental should not be inlined, check presence of elemental and apply.
// CHECK: %[[ELEM:.*]] = hlfir.elemental
// CHECK: fir.do_loop
// CHECK:   fir.store
// CHECK:   %[[APPLIED:.*]] = hlfir.apply %[[ELEM]]
// CHECK:   fir.store %[[APPLIED]]
// Inlined code (coordinate_of/load) should not appear inside the loop.
// CHECK-NOT: fir.coordinate_of %arg0, %arg1


// Check successful inlining where 2-d hlfir.elemental survives because the
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

  // Producer (Elemental).
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

// Check global store blocks inlining.
fir.global @mask_storage : !hlfir.expr<10x10x!fir.logical<4>>
func.func @test_nested_elemental(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10, %c10 : (index, index) -> !fir.shape<2>

  // The Elemental Mask (b * c).
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
// CHECK: %[[MASK:.*]] = hlfir.elemental
// CHECK: %[[PTR:.*]] = fir.address_of(@mask_storage)
// CHECK: fir.store %[[MASK]] to %[[PTR]]
// Apply site not inlined.
// CHECK: fir.do_loop
// CHECK:   fir.do_loop
// CHECK:     %[[VAL:.*]] = hlfir.apply %[[MASK]]
// CHECK:     fir.store %[[VAL]]
// The designate/load should only be inside the elemental.
// CHECK-NOT: hlfir.designate

// Inlining into a single hlfir.apply (relaxed inlining).
// a = (b * c)[1]
func.func @test_scalar_apply_inlining_safe(%b: !fir.ref<!fir.array<10xf32>>, %c1: index) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>

  // Producer(1D Elemental).
  %prod = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ^bb0(%i: index):
    %b_addr = fir.coordinate_of %b, %i : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
    %b_val = fir.load %b_addr : !fir.ref<f32>
    %res = arith.addf %b_val, %b_val : f32
    hlfir.yield_element %res : f32
  }

  // Scalar Apply (Target for inlining).
  %scalar_val = hlfir.apply %prod, %c1 : (!hlfir.expr<10xf32>, index) -> f32

  %temp:3 = hlfir.associate %prod(%shape) : (!hlfir.expr<10xf32>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>, i1)

  hlfir.destroy %prod : !hlfir.expr<10xf32>

  // Use the result.
  %dummy = fir.alloca f32
  fir.store %scalar_val to %dummy : !fir.ref<f32>

  hlfir.end_associate %temp#0, %temp#2 : !fir.ref<!fir.array<10xf32>>, i1
  return
}

// CHECK-LABEL: func.func @test_scalar_apply_inlining_safe
// CHECK: %[[ELEM:.*]] = hlfir.elemental
// CHECK: hlfir.yield_element
// CHECK: %[[ADDR:.*]] = fir.coordinate_of %arg0, %arg1
// CHECK: %[[VAL:.*]] = fir.load %[[ADDR]]
// CHECK: arith.addf %[[VAL]], %[[VAL]]
// CHECK-NOT: hlfir.apply
// CHECK: hlfir.associate %[[ELEM]]
// CHECK-NOT: hlfir.destroy %[[ELEM]]

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

  // tmp1 = a - 1.
  %tmp1 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%i: index):
    %a_ref = hlfir.designate %0#0 (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %a_val = fir.load %a_ref : !fir.ref<f32>
    %sub = arith.subf %a_val, %f1 : f32
    hlfir.yield_element %sub : f32
  }

  %dummy = hlfir.no_reassoc %tmp1 : !hlfir.expr<?xf32>

  // tmp2 = a * tmp1.
  %tmp2 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%j: index):
    %a_ref_2 = hlfir.designate %0#0 (%j) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %t1_val = hlfir.apply %tmp1, %j : (!hlfir.expr<?xf32>, index) -> f32
    %a_val_2 = fir.load %a_ref_2 : !fir.ref<f32>
    %mul = arith.mulf %a_val_2, %t1_val : f32
    hlfir.yield_element %mul : f32
  }

  // tmp3 = sqrt(tmp2).
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

// Test conflicting writes hidden within nested regions (like fir.if) between
// producer and the apply site.
func.func @test_nested_region_conflict(%arg0: !fir.ref<f32>, %cond: i1) {
  %c1 = arith.constant 1 : index
  %shape = fir.shape %c1 : (index)  -> !fir.shape<1>

  // Producer.
  %elem = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<1xf32> {
    ^bb0(%i: index):
      %val = fir.load %arg0 : !fir.ref<f32>
      hlfir.yield_element %val : f32
  }

  // Nested region (fir.if) containing a store.
  // This tests if your walk(func) correctly sees into the 'then' block.
  fir.if %cond {
    %new_val = arith.constant 3.0 : f32
    fir.store %new_val to %arg0 : !fir.ref<f32>
  } else {
    // Distractor op
    fir.no_reassoc %cond : i1
  }

  // Apply Site.
  %res = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<1xf32> {
    ^bb0(%j: index):
      %val = hlfir.apply %elem, %j : (!hlfir.expr<1xf32>, index) -> f32
      hlfir.yield_element %val : f32
  }

  hlfir.destroy %elem : !hlfir.expr<1xf32>
  return
}
// CHECK-LABEL: func.func @test_nested_region_conflict
// CHECK: %[[ELEM:.*]] = hlfir.elemental
// CHECK: fir.if %{{.*}} {
// CHECK:   fir.store
// CHECK: }
// CHECK: hlfir.elemental
// CHECK:   hlfir.apply %[[ELEM]]
// CHECK: hlfir.destroy %[[ELEM]]

// Checks tracking the elemntal result through block arguments to find the
// hlfir.apply site across a branch, fir.store in the intervening block (^bb1)
// blocks the inlining. It was getting inlined with relaxed inlining patch.
func.func @test_cross_block_conflict(%arg0: !fir.ref<f32>, %shape: !fir.shape<1>) {
  // Producer in Entry Block.
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<1xf32> {
  ^bb0(%i: index):
    %val = fir.load %arg0 : !fir.ref<f32>
    hlfir.yield_element %val : f32
  }

  // Pass value to block argument to maintain dataflow for the worklist.
  cf.br ^bb1(%elemental : !hlfir.expr<1xf32>)

^bb1(%block_arg: !hlfir.expr<1xf32>):
  // Conflicting Write.
  // This store between producer and apply site across blocks.
  %new_val = arith.constant 2.0 : f32
  fir.store %new_val to %arg0 : !fir.ref<f32>

  // Apply Site.
  %res = hlfir.elemental %shape : (!fir.shape<1>) -> !hlfir.expr<1xf32> {
  ^bb0(%j: index):
    %val = hlfir.apply %block_arg, %j : (!hlfir.expr<1xf32>, index) -> f32
    hlfir.yield_element %val : f32
  }

  hlfir.destroy %elemental : !hlfir.expr<1xf32>
  return
}
// CHECK-LABEL: func.func @test_cross_block_conflict
// CHECK: %[[ELM:.*]] = hlfir.elemental
// CHECK: cf.br ^bb1(%[[ELM]] : !hlfir.expr<1xf32>)
// CHECK: ^bb1(%[[BARG:.*]]: !hlfir.expr<1xf32>):
// CHECK: fir.store {{.*}} to %arg0
// CHECK: hlfir.elemental
// CHECK: hlfir.apply %[[BARG]]
// CHECK: hlfir.destroy %[[ELM]]

// External impure procedure that might modify %arg0.
func.func private @impure_side_effect()

func.func @test_impure_call_conflict(%arg0: !fir.ref<f32>, %shape: !fir.shape<1>) {
  // Reads from %arg0.
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<1xf32> {
  ^bb0(%i: index):
    %val = fir.load %arg0 : !fir.ref<f32>
    hlfir.yield_element %val : f32
  }

  // Acts as a memory barrier.
  fir.call @impure_side_effect() : () -> ()

  // Apply Site.
  %res = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<1xf32> {
  ^bb0(%j: index):
    %val = hlfir.apply %elemental, %j : (!hlfir.expr<1xf32>, index) -> f32
    hlfir.yield_element %val : f32
  }

  hlfir.destroy %elemental : !hlfir.expr<1xf32>
  return
}
// CHECK-LABEL: func.func @test_impure_call_conflict
// CHECK: %[[ELEM:.*]] = hlfir.elemental
// CHECK: fir.call @impure_side_effect()
// CHECK: hlfir.apply %[[ELEM]]
// CHECK: hlfir.destroy %[[ELEM]]

// Check conflicting write to the same memory buffer read by the elemental
// producer in loop body blocks inlining.
func.func @test_memory_dependency_with_designate(%arg0: !fir.ref<!fir.array<10xf32>>, %shape: !fir.shape<1>) {
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %val = arith.constant 2.0 : f32

  // Reads from the whole array %arg0.
  %elem = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
    ^bb0(%i: index):
      %addr = fir.coordinate_of %arg0, %i : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
      %load = fir.load %addr : !fir.ref<f32>
      hlfir.yield_element %load : f32
  }

  // Modifies one element of the same array, partial write to the same base
  // buffer.
  %specific_addr = hlfir.designate %arg0 (%c1) : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
  fir.store %val to %specific_addr : !fir.ref<f32>

  // Apply Site.
  %res = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
    ^bb0(%j: index):
      %applied = hlfir.apply %elem, %j : (!hlfir.expr<10xf32>, index) -> f32
      hlfir.yield_element %applied : f32
  }

  hlfir.destroy %elem : !hlfir.expr<10xf32>
  return
}
// CHECK-LABEL: func.func @test_memory_dependency_with_designate
// CHECK: %[[ELEM:.*]] = hlfir.elemental {{.*}} unordered
// CHECK: fir.store {{.*}} to %{{.*}}
// CHECK: hlfir.elemental
// CHECK:   hlfir.apply %[[ELEM]]
// CHECK: hlfir.destroy %[[ELEM]]
