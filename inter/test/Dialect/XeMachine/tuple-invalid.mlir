// RUN: not inter-opt %s 2>&1 | FileCheck %s

// CHECK: error: 'xemachine.update_tuple' op offsets must be sorted and non-overlapping
func.func @overlapping_updates(%base: !xemachine.reg<32, -1>,
                               %a: !xemachine.reg<16, -1>,
                               %b: !xemachine.reg<16, -1>) {
  %updated = xemachine.update_tuple %base, %a, %b {offsets = [0, 8]}
      : (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>,
         !xemachine.reg<16, -1>) -> !xemachine.reg<32, -1>
  return
}
