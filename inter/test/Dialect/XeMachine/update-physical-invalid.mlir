// RUN: not inter-opt %s 2>&1 | FileCheck %s

// CHECK: error: 'xemachine.update_tuple' op physical update placement must match its tuple offset
func.func @unaligned_physical_update(%base: !xemachine.reg<32, 4>,
                                     %update: !xemachine.reg<16, 4>) {
  %updated = xemachine.update_tuple %base, %update {offsets = [16]}
      : (!xemachine.reg<32, 4>, !xemachine.reg<16, 4>)
        -> !xemachine.reg<32, 4>
  return
}
