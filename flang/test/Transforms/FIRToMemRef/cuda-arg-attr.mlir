// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @cuda_box_arg
// CHECK:       [[SHIFT:%.*]] = fir.shift
// CHECK:       [[BOX_ADDR:%.*]] = fir.box_addr %{{.*}} {cuf.data_attr = #cuf.cuda<managed>} : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:       fir.convert [[BOX_ADDR]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:       memref.store
func.func @cuda_box_arg(%arg0: !fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<managed>}) {
  %c0 = arith.constant 0 : index
  %shift = fir.shift %c0 : (index) -> !fir.shift<1>
  %box = fir.convert %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %coor = fir.array_coor %box(%shift) %c0 : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, index) -> !fir.ref<f32>
  %cst = arith.constant 0.000000e+00 : f32
  fir.store %cst to %coor : !fir.ref<f32>
  return
}
