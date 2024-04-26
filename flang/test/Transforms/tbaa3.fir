// Test experimental option to add tbaa trees for things that aren't functions

// RUN: fir-opt --nonfunc-tbaa --fir-add-alias-tags %s | FileCheck %s


// CHECK: #[[TBAA_ROOT:.*]] = #llvm.tbaa_root<id = "Flang function root add_reduction_byref_box_3xi32">
// CHECK: #[[ANY_ACCESS:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[TBAA_ROOT]], 0>}>
// CHECK: #[[ANY_DATA:.*]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANY_ACCESS]], 0>}>
// CHECK: #[[DIRECT_DATA:.*]] = #llvm.tbaa_type_desc<id = "direct data", members = {<#[[ANY_DATA]], 0>}>
// CHECK: #[[GLOBAL_A:.*]] = #llvm.tbaa_type_desc<id = "direct data/_QMmodEa", members = {<#[[DIRECT_DATA]], 0>}>
// CHECK: #[[GLOBAL_A_TAG:.*]] = #llvm.tbaa_tag<base_type = #[[GLOBAL_A]], access_type = #[[GLOBAL_A]], offset = 0>

fir.global @_QMmodEa : !fir.box<!fir.heap<!fir.array<?xi32>>> {
  %c0 = arith.constant 0 : index
  %0 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %1 = fir.shape %c0 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xi32>>>
}

// nonsense reduction to give us something that we can generate TBAA tags for
omp.declare_reduction @add_reduction_byref_box_3xi32 : !fir.ref<!fir.box<!fir.array<3xi32>>> init {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<3xi32>>>):
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
  %1 = fir.address_of(@_QMmodEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %c3 = arith.constant 3 : index
  %2 = fir.shape %c3 : (index) -> !fir.shape<1>
  %3 = fir.allocmem !fir.array<3xi32> {bindc_name = ".tmp", uniq_name = ""}
  %true = arith.constant true
  %4:2 = hlfir.declare %3(%2) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<3xi32>>, !fir.heap<!fir.array<3xi32>>)
  %c0 = arith.constant 0 : index
  %5:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.array<3xi32>>, index) -> (index, index, index)
  %6 = fir.shape_shift %5#0, %5#1 : (index, index) -> !fir.shapeshift<1>
  %7 = fir.embox %4#0(%6) : (!fir.heap<!fir.array<3xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<3xi32>>
  hlfir.assign %c0_i32 to %7 : i32, !fir.box<!fir.array<3xi32>>
  %8 = fir.convert %7 : (!fir.box<!fir.array<3xi32>>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %8 to %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %9 = fir.convert %1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.array<3xi32>>>
  omp.yield(%9 : !fir.ref<!fir.box<!fir.array<3xi32>>>)
} combiner {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<3xi32>>>, %arg1: !fir.ref<!fir.box<!fir.array<3xi32>>>):
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
  %1 = fir.address_of(@_QMmodEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %a = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %c0 = arith.constant 0 : index
  %2:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.array<3xi32>>, index) -> (index, index, index)
  %3 = fir.shape_shift %2#0, %2#1 : (index, index) -> !fir.shapeshift<1>
  %c1 = arith.constant 1 : index
  fir.do_loop %arg2 = %c1 to %2#1 step %c1 unordered {
    %4 = fir.array_coor %0(%3) %arg2 : (!fir.box<!fir.array<3xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
    %5 = fir.array_coor %a(%3) %arg2 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
    %6 = fir.load %4 : !fir.ref<i32>
    %7 = fir.load %5 : !fir.ref<i32>
// CHECK:      fir.load %{{.*}} {tbaa = [#[[GLOBAL_A_TAG]]]} : !fir.ref<i32>
    %8 = arith.addi %6, %7 : i32
    fir.store %8 to %4 : !fir.ref<i32>
  }
  omp.yield(%arg0 : !fir.ref<!fir.box<!fir.array<3xi32>>>)
}  cleanup {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<3xi32>>>):
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
  %1 = fir.box_addr %0 : (!fir.box<!fir.array<3xi32>>) -> !fir.ref<!fir.array<3xi32>>
  %2 = fir.convert %1 : (!fir.ref<!fir.array<3xi32>>) -> i64
  %c0_i64 = arith.constant 0 : i64
  %3 = arith.cmpi ne, %2, %c0_i64 : i64
  fir.if %3 {
    %4 = fir.convert %1 : (!fir.ref<!fir.array<3xi32>>) -> !fir.heap<!fir.array<3xi32>>
    fir.freemem %4 : !fir.heap<!fir.array<3xi32>>
  }
  omp.yield
}
