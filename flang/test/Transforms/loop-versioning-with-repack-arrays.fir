// RUN: fir-opt --loop-versioning %s | FileCheck %s

// Check that LoopVersioning kicks in when there is fir.pack_array
// in between fir.declare and the block argument.

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QPtest(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.pack_array %arg0 stack whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
    %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtestEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
    %3 = fir.rebox %2 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
    %4:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
    fir.do_loop %arg1 = %c1 to %4#1 step %c1 unordered {
      %5 = fir.array_coor %2 %arg1 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      fir.store %cst to %5 : !fir.ref<f32>
    }
    fir.unpack_array %1 to %arg0 stack : !fir.box<!fir.array<?xf32>>
    return
  }
}
// CHECK-LABEL:   func.func @_QPtest(
// CHECK-SAME:                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_5:.*]] = fir.pack_array %[[VAL_0]] stack whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
// CHECK:           %[[VAL_6:.*]] = fir.declare %[[VAL_5]] dummy_scope %{{.*}} {uniq_name = "_QFtestEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
// CHECK:           %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_6]], %{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_11:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_10]]#2, %[[VAL_11]] : index
// CHECK:           fir.if %[[VAL_12]] {
// CHECK:             fir.do_loop {{.*}} {
// CHECK:               fir.coordinate_of {{.*}} : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:             }
// CHECK:           } else {
// CHECK:             fir.do_loop {{.*}} {
// CHECK:               fir.array_coor {{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:             }
// CHECK:           }
// CHECK:           fir.unpack_array %[[VAL_5]] to %[[VAL_0]] stack : !fir.box<!fir.array<?xf32>>
