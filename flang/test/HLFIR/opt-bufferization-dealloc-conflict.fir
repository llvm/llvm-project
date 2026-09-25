// RUN: fir-opt --opt-bufferization %s | FileCheck %s

// Verify that ElementalAssignBufferization does NOT fuse an elemental into the
// assignment when a deallocation between the elemental and the assignment frees
// memory that the elemental reads. Moving the elemental evaluation down to the
// assignment would read freed memory.

func.func @dealloc_conflict(%dst: !fir.box<!fir.array<?xf32>>, %n: index) {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : f32
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  %heap = fir.allocmem !fir.array<?xf32>, %n {uniq_name = ".src"}
  %src = fir.embox %heap(%shape) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  %elem = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%i: index):
    %d = hlfir.designate %src (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %v = fir.load %d : !fir.ref<f32>
    %s = arith.addf %v, %cst fastmath<contract> : f32
    hlfir.yield_element %s : f32
  }
  // Deallocation of the elemental's read source between elemental and assign.
  fir.freemem %heap : !fir.heap<!fir.array<?xf32>>
  hlfir.assign %elem to %dst : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.destroy %elem : !hlfir.expr<?xf32>
  return
}

// CHECK-LABEL: func.func @dealloc_conflict
// The elemental and the array-level assign must be preserved (no fusion).
// CHECK: hlfir.elemental
// CHECK: fir.freemem
// CHECK: hlfir.assign %{{.*}} to %{{.*}} : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>

// Same conflict through cuf.free, which frees the memory of its operand.
func.func @cuf_dealloc_conflict(%dst: !fir.box<!fir.array<?xf32>>, %n: index) {
  %cst = arith.constant 1.000000e+00 : f32
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  %ptr = cuf.alloc !fir.array<?xf32>, %n : index {bindc_name = ".src", data_attr = #cuf.cuda<managed>, uniq_name = ".src"} -> !fir.ref<!fir.array<?xf32>>
  %heap = fir.convert %ptr : (!fir.ref<!fir.array<?xf32>>) -> !fir.heap<!fir.array<?xf32>>
  %src = fir.embox %heap(%shape) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  %elem = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%i: index):
    %d = hlfir.designate %src (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %v = fir.load %d : !fir.ref<f32>
    %s = arith.addf %v, %cst fastmath<contract> : f32
    hlfir.yield_element %s : f32
  }
  cuf.free %ptr : !fir.ref<!fir.array<?xf32>> {data_attr = #cuf.cuda<managed>}
  hlfir.assign %elem to %dst : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.destroy %elem : !hlfir.expr<?xf32>
  return
}

// CHECK-LABEL: func.func @cuf_dealloc_conflict
// CHECK: hlfir.elemental
// CHECK: cuf.free
// CHECK: hlfir.assign %{{.*}} to %{{.*}} : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>

// A cuf.free of memory that the elemental does not read must not block the
// fusion: the free effect is attached to the freed pointer, so the alias
// analysis can tell the two apart.
func.func @cuf_dealloc_no_conflict(%dst: !fir.box<!fir.array<?xf32>>, %src: !fir.box<!fir.array<?xf32>>, %n: index) {
  %cst = arith.constant 1.000000e+00 : f32
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  %ptr = cuf.alloc !fir.array<?xf32>, %n : index {bindc_name = ".unrelated", data_attr = #cuf.cuda<managed>, uniq_name = ".unrelated"} -> !fir.ref<!fir.array<?xf32>>
  %elem = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%i: index):
    %d = hlfir.designate %src (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %v = fir.load %d : !fir.ref<f32>
    %s = arith.addf %v, %cst fastmath<contract> : f32
    hlfir.yield_element %s : f32
  }
  cuf.free %ptr : !fir.ref<!fir.array<?xf32>> {data_attr = #cuf.cuda<managed>}
  hlfir.assign %elem to %dst : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.destroy %elem : !hlfir.expr<?xf32>
  return
}

// CHECK-LABEL: func.func @cuf_dealloc_no_conflict
// The elemental is fused into the assignment loop.
// CHECK-NOT: hlfir.elemental
// CHECK: fir.do_loop
// CHECK: hlfir.assign %{{.*}} to %{{.*}} : f32, !fir.ref<f32>
