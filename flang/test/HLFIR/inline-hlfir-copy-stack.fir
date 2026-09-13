// Test the stack/heap placement of the buffer created when inlining
// hlfir.copy_in. Only buffers whose size is a compile-time constant not
// exceeding the threshold are placed on the stack.

// RUN: fir-opt --inline-hlfir-copy="small-array-threshold=64" %s \
// RUN:   | FileCheck %s
// RUN: fir-opt --inline-hlfir-copy="small-array-threshold=1024" %s \
// RUN:   | FileCheck %s --check-prefix=BIGTHRESHOLD

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"} {

func.func private @callee8(!fir.ref<!fir.array<8xf64>>) -> ()
func.func private @callee100(!fir.ref<!fir.array<100xf64>>) -> ()
func.func private @calleeDyn(!fir.ref<!fir.array<?xf64>>) -> ()

// 8 * 8 = 64 bytes, at the threshold: the buffer goes on the stack.
func.func @small_constant_buffer(%arg0: !fir.box<!fir.array<?x?xf64>>, %arg1: !fir.ref<i32>) {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<8xf64>>>
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %1 = fir.load %arg1 : !fir.ref<i32>
  %2 = fir.convert %1 : (i32) -> i64
  %3 = fir.shape %c8 : (index) -> !fir.shape<1>
  %4 = hlfir.designate %arg0 (%2, %c1:%c8:%c1)  shape %3 : (!fir.box<!fir.array<?x?xf64>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<8xf64>>
  %5:2 = hlfir.copy_in %4 to %0 : (!fir.box<!fir.array<8xf64>>, !fir.ref<!fir.box<!fir.heap<!fir.array<8xf64>>>>) -> (!fir.box<!fir.array<8xf64>>, i1)
  %6 = fir.box_addr %5#0 : (!fir.box<!fir.array<8xf64>>) -> !fir.ref<!fir.array<8xf64>>
  fir.call @callee8(%6) : (!fir.ref<!fir.array<8xf64>>) -> ()
  hlfir.copy_out %0, %5#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<8xf64>>>>, i1) -> ()
  return
}

// 100 * 8 = 800 bytes, above the 64 byte threshold: the buffer stays on the
// heap and is freed by the inlined copy-out.
func.func @big_constant_buffer(%arg0: !fir.box<!fir.array<?x?xf64>>, %arg1: !fir.ref<i32>) {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<100xf64>>>
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %1 = fir.load %arg1 : !fir.ref<i32>
  %2 = fir.convert %1 : (i32) -> i64
  %3 = fir.shape %c100 : (index) -> !fir.shape<1>
  %4 = hlfir.designate %arg0 (%2, %c1:%c100:%c1)  shape %3 : (!fir.box<!fir.array<?x?xf64>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<100xf64>>
  %5:2 = hlfir.copy_in %4 to %0 : (!fir.box<!fir.array<100xf64>>, !fir.ref<!fir.box<!fir.heap<!fir.array<100xf64>>>>) -> (!fir.box<!fir.array<100xf64>>, i1)
  %6 = fir.box_addr %5#0 : (!fir.box<!fir.array<100xf64>>) -> !fir.ref<!fir.array<100xf64>>
  fir.call @callee100(%6) : (!fir.ref<!fir.array<100xf64>>) -> ()
  hlfir.copy_out %0, %5#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<100xf64>>>>, i1) -> ()
  return
}

// Runtime extent: the buffer stays on the heap whatever the threshold is,
// because a dynamically sized fir.alloca cannot be hoisted out of the copy-in
// branch and would grow the stack if the copy-in were inside a loop.
func.func @dynamic_buffer(%arg0: !fir.box<!fir.array<?x?xf64>>, %arg1: !fir.ref<i32>, %n: index) {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf64>>>
  %c1 = arith.constant 1 : index
  %1 = fir.load %arg1 : !fir.ref<i32>
  %2 = fir.convert %1 : (i32) -> i64
  %3 = fir.shape %n : (index) -> !fir.shape<1>
  %4 = hlfir.designate %arg0 (%2, %c1:%n:%c1)  shape %3 : (!fir.box<!fir.array<?x?xf64>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
  %5:2 = hlfir.copy_in %4 to %0 : (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.box<!fir.array<?xf64>>, i1)
  %6 = fir.box_addr %5#0 : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
  fir.call @calleeDyn(%6) : (!fir.ref<!fir.array<?xf64>>) -> ()
  hlfir.copy_out %0, %5#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, i1) -> ()
  return
}

}

// The buffer is created in the copy-in branch, but being a constant size
// fir.alloca it is hoisted to the entry block, and no deallocation code is
// generated for the copy-out.

// CHECK-LABEL:   func.func @small_constant_buffer(
// CHECK:           %[[BUF:.*]] = fir.alloca !fir.array<8xf64> {bindc_name = ".tmp.copy_in"}
// CHECK:           %[[SECTION:.*]] = hlfir.designate %{{.*}} shape
// CHECK:           %[[CONTIG:.*]] = fir.is_contiguous_box %[[SECTION]]
// CHECK:           %[[RES:.*]]:2 = fir.if %[[CONTIG]] -> (!fir.box<!fir.array<8xf64>>, i1) {
// CHECK:             fir.result %[[SECTION]], %false : !fir.box<!fir.array<8xf64>>, i1
// CHECK:           } else {
// CHECK:             %[[DECL:.*]]:2 = hlfir.declare %[[BUF]]
// CHECK:             fir.do_loop
// CHECK:             %[[EMBOX:.*]] = fir.embox %[[DECL]]#0
// CHECK:             fir.result %[[EMBOX]], %true : !fir.box<!fir.array<8xf64>>, i1
// CHECK:           }
// CHECK:           %[[ADDR:.*]] = fir.box_addr %[[RES]]#0
// CHECK:           fir.call @callee8(%[[ADDR]])
// CHECK-NOT:       fir.allocmem
// CHECK-NOT:       fir.freemem

// Above the threshold: the buffer is allocated in the copy-in branch and freed
// under the flag yielded by the fir.if. The free reads the address from the box
// yielded by the fir.if, which needs no spill to memory.

// CHECK-LABEL:   func.func @big_constant_buffer(
// CHECK:           %[[RES:.*]]:2 = fir.if %{{.*}} -> (!fir.box<!fir.array<100xf64>>, i1) {
// CHECK:             fir.result %{{.*}}, %false : !fir.box<!fir.array<100xf64>>, i1
// CHECK:           } else {
// CHECK:             fir.allocmem !fir.array<100xf64> {bindc_name = ".tmp.copy_in"
// CHECK:             fir.result %{{.*}}, %true : !fir.box<!fir.array<100xf64>>, i1
// CHECK:           }
// CHECK:           fir.call @callee100
// CHECK:           fir.if %[[RES]]#1 {
// CHECK:             %[[FREE_ADDR:.*]] = fir.box_addr %[[RES]]#0
// CHECK:             %[[FREE_HEAP:.*]] = fir.convert %[[FREE_ADDR]]
// CHECK:             fir.freemem %[[FREE_HEAP]]

// A runtime extent always stays on the heap.

// CHECK-LABEL:   func.func @dynamic_buffer(
// CHECK:           %[[RES:.*]]:2 = fir.if %{{.*}} -> (!fir.box<!fir.array<?xf64>>, i1) {
// CHECK:           } else {
// CHECK:             fir.allocmem !fir.array<?xf64>, %{{.*}} {bindc_name = ".tmp.copy_in"
// CHECK:           }
// CHECK:           fir.call @calleeDyn
// CHECK:           fir.if %[[RES]]#1 {
// CHECK:             fir.freemem

// With a threshold that covers it, the 800 byte buffer is also placed on the
// stack, while the runtime-sized one is not.

// BIGTHRESHOLD-LABEL:   func.func @big_constant_buffer(
// BIGTHRESHOLD:           %[[BUF:.*]] = fir.alloca !fir.array<100xf64> {bindc_name = ".tmp.copy_in"}
// BIGTHRESHOLD:           fir.if %{{.*}} -> (!fir.box<!fir.array<100xf64>>, i1) {
// BIGTHRESHOLD:             hlfir.declare %[[BUF]]
// BIGTHRESHOLD:           fir.call @callee100
// BIGTHRESHOLD-NOT:       fir.allocmem
// BIGTHRESHOLD-NOT:       fir.freemem

// BIGTHRESHOLD-LABEL:   func.func @dynamic_buffer(
// BIGTHRESHOLD:           fir.allocmem !fir.array<?xf64>, %{{.*}} {bindc_name = ".tmp.copy_in"
// BIGTHRESHOLD:           fir.freemem
