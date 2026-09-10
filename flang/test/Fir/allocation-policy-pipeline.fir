// Test that the pass creating a copy-in buffer and the pass placing existing
// allocations agree, because both read the policy recorded on the module. If
// they disagreed, a buffer put on the stack by the first would be moved back to
// the heap by the second (or the other way round), and the placement would
// depend on the order the passes happen to run in.

// A 64 byte threshold puts the 800 byte buffer on the heap, and it must stay
// there.
// RUN: fir-opt --inline-hlfir-copy --allocation-placement %s \
// RUN:   | FileCheck %s --check-prefix=SMALL

// A 4096 byte threshold puts it on the stack, and it must stay there.
// RUN: fir-opt --inline-hlfir-copy="small-array-threshold=4096" \
// RUN:   --allocation-placement="small-array-threshold=4096" %s \
// RUN:   | FileCheck %s --check-prefix=BIG

module attributes {fir.allocation_policy =
                       #fir.allocation_policy<stack_arrays = false,
                           small_array_threshold = 64,
                           total_stack_limit = 4194304>,
                   fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "",
                   llvm.data_layout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"} {

func.func private @callee100(!fir.ref<!fir.array<100xf64>>) -> ()

// SMALL-LABEL:   func.func @copy_in_buffer
// SMALL:           fir.allocmem !fir.array<100xf64> {bindc_name = ".tmp.copy_in"
// SMALL:           fir.freemem
// SMALL-NOT:       fir.alloca !fir.array<100xf64>

// BIG-LABEL:     func.func @copy_in_buffer
// BIG:             fir.alloca !fir.array<100xf64> {bindc_name = ".tmp.copy_in"
// BIG-NOT:         fir.allocmem
// BIG-NOT:         fir.freemem
func.func @copy_in_buffer(%arg0: !fir.box<!fir.array<?x?xf64>>, %arg1: !fir.ref<i32>) {
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

}
