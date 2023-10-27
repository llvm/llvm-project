// Test hlfir.get_extent operaiton parse, verify (no errors), and unparse
// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @getextent(%arg0: !fir.shape<3>) {
  %0 = hlfir.get_extent %arg0 {dim = 0 : index} : (!fir.shape<3>) -> index
  %1 = hlfir.get_extent %arg0 {dim = 1 : index} : (!fir.shape<3>) -> index
  %2 = hlfir.get_extent %arg0 {dim = 2 : index} : (!fir.shape<3>) -> index
  return
}
// CHECK-LABEL: func.func @getextent
// CHECK:           %[[SHAPE:.*]]: !fir.shape<3>
// CHECK-NEXT:    %[[EXT0:.*]] = hlfir.get_extent %[[SHAPE]] {dim = 0 : index} : (!fir.shape<3>) -> index
// CHECK-NEXT:    %[[EXT1:.*]] = hlfir.get_extent %[[SHAPE]] {dim = 1 : index} : (!fir.shape<3>) -> index
// CHECK-NEXT:    %[[EXT2:.*]] = hlfir.get_extent %[[SHAPE]] {dim = 2 : index} : (!fir.shape<3>) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
