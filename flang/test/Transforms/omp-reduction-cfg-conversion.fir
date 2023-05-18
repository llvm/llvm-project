// RUN: fir-opt --cfg-conversion-on-reduce-opt %s | FileCheck %s

omp.declare_reduction @add_reduction_i_32_box_3_byref : !fir.ref<!fir.box<!fir.array<3xi32>>> init {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<3xi32>>>):
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  %c3 = arith.constant 3 : index
  %0 = fir.alloca !fir.box<!fir.array<3xi32>>
  %1 = fir.alloca !fir.array<3xi32> {bindc_name = "omp.reduction.array.init"}
  %2 = fir.shape %c3 : (index) -> !fir.shape<1>
  %3 = fir.declare %1(%2) {uniq_name = "omp.reduction.array.init"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<3xi32>>
  %4 = fir.embox %3(%2) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
  %5 = fir.alloca i32
  fir.store %c0_i32 to %5 : !fir.ref<i32>
  %6 = fir.embox %5 : (!fir.ref<i32>) -> !fir.box<i32>
  fir.store %4 to %0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
  %7 = fir.address_of(@_QQclX9a9fdf8c5fd329fbbf2b0c08e2ca9a1e) : !fir.ref<!fir.char<1,40>>
  %8 = fir.convert %0 : (!fir.ref<!fir.box<!fir.array<3xi32>>>) -> !fir.ref<!fir.box<none>>
  %9 = fir.convert %6 : (!fir.box<i32>) -> !fir.box<none>
  %10 = fir.convert %7 : (!fir.ref<!fir.char<1,40>>) -> !fir.ref<i8>
  %11 = fir.call @_FortranAAssign(%8, %9, %10, %c4_i32) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  %12 = fir.alloca !fir.box<!fir.array<3xi32>>
  fir.store %4 to %12 : !fir.ref<!fir.box<!fir.array<3xi32>>>
  omp.yield(%12 : !fir.ref<!fir.box<!fir.array<3xi32>>>)
} combiner {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<3xi32>>>, %arg1: !fir.ref<!fir.box<!fir.array<3xi32>>>):
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
  %1 = fir.load %arg1 : !fir.ref<!fir.box<!fir.array<3xi32>>>
  %2:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.array<3xi32>>, index) -> (index, index, index)
  %3 = fir.shape_shift %2#0, %2#1 : (index, index) -> !fir.shapeshift<1>
  fir.do_loop %arg2 = %c1 to %2#1 step %c1 unordered {
    %4 = fir.array_coor %0(%3) %arg2 : (!fir.box<!fir.array<3xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
    %5 = fir.array_coor %1(%3) %arg2 : (!fir.box<!fir.array<3xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
    %6 = fir.load %4 : !fir.ref<i32>
    %7 = fir.load %5 : !fir.ref<i32>
    %8 = arith.addi %6, %7 : i32
    fir.store %8 to %4 : !fir.ref<i32>
  }
  omp.yield(%arg0 : !fir.ref<!fir.box<!fir.array<3xi32>>>)
}

// ensure cfg conversion has run on the do loop
// CHECK: combiner {
// CHECK-NOT: fir.do_loop
// CHECK: ^bb0({{.*}}):
// ...
// CHECK:   cf.br ^bb1
// CHECK: ^bb1({{.*}}):
// ...
// CHECK:   cf.cond_br %{{.*}} ^bb2, ^bb3
// CHECK: ^bb2:
// ...
// CHECK:   cf.br ^bb1
// CHECK: ^bb3:

