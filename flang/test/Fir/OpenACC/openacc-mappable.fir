// Use --mlir-disable-threading so that the diagnostic printing is serialized.
// RUN: fir-opt %s -pass-pipeline='builtin.module(test-fir-openacc-interfaces)' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<f16 = dense<16> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
  func.func @_QPsub() {
    %c2 = arith.constant 2 : index
    %c10 = arith.constant 10 : index
    %0 = fir.alloca !fir.array<10xf32> {bindc_name = "arr", uniq_name = "_QFsubEarr"}
    %1 = fir.shape_shift %c2, %c10 : (index, index) -> !fir.shapeshift<1>
    %2 = fir.declare %0(%1) {uniq_name = "_QFsubEarr"} : (!fir.ref<!fir.array<10xf32>>, !fir.shapeshift<1>) -> !fir.ref<!fir.array<10xf32>>
    %3 = fir.embox %2(%1) : (!fir.ref<!fir.array<10xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<10xf32>>
    %4 = fir.box_addr %3 : (!fir.box<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>>
    %5 = acc.copyin var(%3 : !fir.box<!fir.array<10xf32>>) -> !fir.box<!fir.array<10xf32>> {name = "arr", structured = false}
    %6 = acc.copyin varPtr(%4 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "arr", structured = false}
    acc.enter_data dataOperands(%5, %6 : !fir.box<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
    return
  }
}

// CHECK: Visiting: %{{.*}} = acc.copyin var(%{{.*}} : !fir.box<!fir.array<10xf32>>) -> !fir.box<!fir.array<10xf32>> {name = "arr", structured = false}
// CHECK: Mappable: !fir.box<!fir.array<10xf32>>
// CHECK: Size: 40
// CHECK: Visiting: %{{.*}} = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "arr", structured = false}
// CHECK: Mappable: !fir.array<10xf32>
// CHECK: Size: 40
