// RUN: fir-opt %s -acc-recipe-materialization | FileCheck %s

// A descriptor holds the mapped storage itself, so a section's bounds describe
// the payload it points to and must not reach the initial-value map.

module {
  acc.private.recipe @privatization_box_heap : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> init {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>):
    %c0 = arith.constant 0 : index
    %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %1:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    %2 = fir.allocmem !fir.array<?xf32>, %1#1
    %3 = fir.shape_shift %1#0, %1#1 : (index, index) -> !fir.shapeshift<1>
    %4 = fir.embox %2(%3) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    %5 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.store %4 to %5 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    acc.yield %5 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  }

  acc.firstprivate.recipe @firstprivatization_box_heap : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> init {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>):
    %c0 = arith.constant 0 : index
    %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %1:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    %2 = fir.allocmem !fir.array<?xf32>, %1#1
    %3 = fir.shape_shift %1#0, %1#1 : (index, index) -> !fir.shapeshift<1>
    %4 = fir.embox %2(%3) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    %5 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.store %4 to %5 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    acc.yield %5 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  } copy {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>,
       %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>):
    acc.terminator
  }

  acc.firstprivate.recipe @firstprivatization_ref_array : !fir.ref<!fir.array<?xf32>> init {
  ^bb0(%arg0: !fir.ref<!fir.array<?xf32>>):
    %c89 = arith.constant 89 : index
    %0 = fir.alloca !fir.array<?xf32>, %c89
    acc.yield %0 : !fir.ref<!fir.array<?xf32>>
  } copy {
  ^bb0(%arg0: !fir.ref<!fir.array<?xf32>>, %arg1: !fir.ref<!fir.array<?xf32>>):
    acc.terminator
  }

  // A private descriptor with an array section: the map takes no bounds.
  func.func @private_desc_section(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c88 = arith.constant 88 : index
    %c89 = arith.constant 89 : index
    %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c88 : index) extent(%c89 : index) stride(%c4 : index) startIdx(%c1 : index) strideInBytes(true)
    %priv = acc.private varPtr(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) bounds(%bounds) recipe(@privatization_box_heap) name("t(2:90)") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    acc.parallel private(%priv : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
      %0 = fir.load %priv : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
      acc.yield
    }
    return
  }

  // The same holds for firstprivate, which shares the initial-value mapping.
  func.func @firstprivate_desc_section(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c88 = arith.constant 88 : index
    %c89 = arith.constant 89 : index
    %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c88 : index) extent(%c89 : index) stride(%c4 : index) startIdx(%c1 : index) strideInBytes(true)
    %priv = acc.firstprivate varPtr(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) bounds(%bounds) recipe(@firstprivatization_box_heap) name("t(2:90)") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    acc.parallel firstprivate(%priv : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
      %0 = fir.load %priv : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
      acc.yield
    }
    return
  }

  // Without a descriptor the bounds do describe the mapped storage, so they
  // are still forwarded.
  func.func @firstprivate_raw_section(%arg0: !fir.ref<!fir.array<?xf32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c88 = arith.constant 88 : index
    %c89 = arith.constant 89 : index
    %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c88 : index) extent(%c89 : index) stride(%c4 : index) startIdx(%c1 : index) strideInBytes(true)
    %priv = acc.firstprivate varPtr(%arg0 : !fir.ref<!fir.array<?xf32>>) bounds(%bounds) recipe(@firstprivatization_ref_array) name("a(2:90)") -> !fir.ref<!fir.array<?xf32>>
    acc.parallel firstprivate(%priv : !fir.ref<!fir.array<?xf32>>) {
      acc.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @private_desc_section
// CHECK: acc.firstprivate_map varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) name("t(2:90)")

// CHECK-LABEL: func.func @firstprivate_desc_section
// CHECK: acc.firstprivate_map varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) name("t(2:90)")

// CHECK-LABEL: func.func @firstprivate_raw_section
// CHECK: acc.firstprivate_map varPtr(%{{.*}} : !fir.ref<!fir.array<?xf32>>) bounds(%{{.*}}) name("a(2:90)")
