// RUN: not inter-opt %s 2>&1 | FileCheck %s

// CHECK: error: 'xemachine.tuple_from_elements' op physical element placement must match its tuple offset
func.func @noncontiguous_tuple(%lo: !xemachine.reg<32, 4>,
                               %hi: !xemachine.reg<32, 7>) {
  %tuple = xemachine.tuple_from_elements %lo, %hi
      : (!xemachine.reg<32, 4>, !xemachine.reg<32, 7>)
        -> !xemachine.reg<64, 4>
  return
}
