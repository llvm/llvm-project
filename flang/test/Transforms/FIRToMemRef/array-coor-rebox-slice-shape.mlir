// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// Descriptor-backed array_coor with fir.shape (extents only, no shift) and a
// rebox slice must rebuild the memref view from fir.box_dims strides. This
// mirrors assign-inlined loops for VP(:,K) on a rebox-sourced box.
//
// Without useBoxDescriptorLayout, FIRToMemRef used fir.shape extents to
// synthesize strides and mis-addressed column sections of non-contiguous boxes.

// CHECK-LABEL: func.func @array_coor_box_slice_shape_no_shift
// CHECK:       fir.box_addr %[[BOX:arg[0-9]+]]
// CHECK:       fir.box_elesize %[[BOX]]
// CHECK:       fir.box_dims %[[BOX]]
// CHECK:       arith.divsi
// CHECK:       fir.box_dims %[[BOX]]
// CHECK:       arith.divsi
// CHECK:       memref.reinterpret_cast %{{.+}} to offset:
// CHECK-SAME:  sizes: [{{%.+}}, {{%.+}}], strides: [{{%.+}}, {{%.+}}]
// CHECK:       memref.load %{{.+}}[{{%.+}}, {{%.+}}]
// CHECK-NOT:   fir.array_coor
func.func @array_coor_box_slice_shape_no_shift(%arg0: !fir.box<!fir.array<?x?xi32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %undef = fir.undefined index
  %rebox = fir.rebox %arg0 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %slice = fir.slice %c1, %c2, %c1, %c2, %undef, %undef :
      (index, index, index, index, index, index) -> !fir.slice<2>
  %dim0:3 = fir.box_dims %rebox, %c0 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  %dim1:3 = fir.box_dims %rebox, %c1 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  %shape = fir.shape %dim0#1, %dim1#1 : (index, index) -> !fir.shape<2>
  %addr = fir.array_coor %arg0(%shape) [%slice] %c1, %c2 :
      (!fir.box<!fir.array<?x?xi32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
  %val = fir.load %addr : !fir.ref<i32>
  return
}

// array_coor on a rebox with explicit fir.shape but NO slice on the
// array_coor itself. The rebox may be non-contiguous, so the descriptor
// owns the layout: extents and strides must come from fir.box_dims, not
// from the synthesized contiguous strides of the explicit shape.
// CHECK-LABEL: func.func @array_coor_rebox_shape_no_slice
// CHECK:       %[[REBOX:.+]] = fir.rebox
// CHECK:       fir.box_addr %[[REBOX]]
// CHECK:       fir.box_elesize %[[REBOX]]
// CHECK:       fir.box_dims %[[REBOX]]
// CHECK:       arith.divsi
// CHECK:       fir.box_dims %[[REBOX]]
// CHECK:       arith.divsi
// CHECK:       memref.reinterpret_cast %{{.+}} to offset:
// CHECK-SAME:  sizes: [{{%.+}}, {{%.+}}], strides: [{{%.+}}, {{%.+}}]
// CHECK:       memref.load %{{.+}}[{{%.+}}, {{%.+}}]
// CHECK-NOT:   fir.array_coor
func.func @array_coor_rebox_shape_no_slice(%arg0: !fir.box<!fir.array<?x?xi32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %rebox = fir.rebox %arg0 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %dim0:3 = fir.box_dims %rebox, %c0 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  %dim1:3 = fir.box_dims %rebox, %c1 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  %shape = fir.shape %dim0#1, %dim1#1 : (index, index) -> !fir.shape<2>
  %addr = fir.array_coor %rebox(%shape) %c1, %c2 :
      (!fir.box<!fir.array<?x?xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
  %val = fir.load %addr : !fir.ref<i32>
  return
}

// Row loop: load VP(I,2) for I=1..2 using the same slice+shape pattern.
// CHECK-LABEL: func.func @array_coor_box_slice_shape_row_loop
// CHECK:       scf.for
// CHECK:         fir.box_elesize %[[BOX:arg[0-9]+]]
// CHECK:         fir.box_dims %[[BOX]]
// CHECK:         arith.divsi
// CHECK:         fir.box_dims %[[BOX]]
// CHECK:         arith.divsi
// CHECK:         memref.reinterpret_cast
// CHECK-SAME:        sizes: [{{%.+}}, {{%.+}}], strides: [{{%.+}}, {{%.+}}]
// CHECK:         memref.load %{{.+}}[{{%.+}}, {{%.+}}]
// CHECK-NOT:     fir.array_coor
func.func @array_coor_box_slice_shape_row_loop(%arg0: !fir.box<!fir.array<?x?xi32>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %undef = fir.undefined index
  %rebox = fir.rebox %arg0 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %slice = fir.slice %c1, %c2, %c1, %c2, %undef, %undef :
      (index, index, index, index, index, index) -> !fir.slice<2>
  %c0 = arith.constant 0 : index
  %trip = arith.subi %c2, %c1 : index
  %ub = arith.addi %trip, %c1 : index
  scf.for %i = %c0 to %ub step %c1 {
    %row = arith.addi %c1, %i : index
    %dim0:3 = fir.box_dims %rebox, %c0 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
    %dim1:3 = fir.box_dims %rebox, %c1 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
    %shape = fir.shape %dim0#1, %dim1#1 : (index, index) -> !fir.shape<2>
    %addr = fir.array_coor %arg0(%shape) [%slice] %row, %c2 :
        (!fir.box<!fir.array<?x?xi32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
    %val = fir.load %addr : !fir.ref<i32>
  }
  return
}
