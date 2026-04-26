// Test insertion of llvm.lifetime for allocmem turn into alloca with constant size.
// RUN: fir-opt --stack-arrays -stack-arrays-lifetime %s | FileCheck %s

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {

func.func @_QPcst_alloca(%arg0: !fir.ref<!fir.array<100000xf32>> {fir.bindc_name = "x"}) {
  %c1 = arith.constant 1 : index
  %c100000 = arith.constant 100000 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c100000 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFcst_allocaEx"} : (!fir.ref<!fir.array<100000xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<100000xf32>>
  %3 = fir.allocmem !fir.array<100000xf32> {bindc_name = ".tmp.array", uniq_name = ""}
  %4 = fir.declare %3(%1) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<100000xf32>>, !fir.shape<1>) -> !fir.heap<!fir.array<100000xf32>>
  fir.do_loop %arg1 = %c1 to %c100000 step %c1 unordered {
    %9 = fir.array_coor %2(%1) %arg1 : (!fir.ref<!fir.array<100000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    %10 = fir.load %9 : !fir.ref<f32>
    %11 = arith.addf %10, %10 fastmath<contract> : f32
    %12 = fir.array_coor %4(%1) %arg1 : (!fir.heap<!fir.array<100000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    fir.store %11 to %12 : !fir.ref<f32>
  }
  %5 = fir.convert %4 : (!fir.heap<!fir.array<100000xf32>>) -> !fir.ref<!fir.array<100000xf32>>
  fir.call @_QPbar(%5) fastmath<contract> : (!fir.ref<!fir.array<100000xf32>>) -> ()
  fir.freemem %4 : !fir.heap<!fir.array<100000xf32>>
  %6 = fir.allocmem !fir.array<100000xi32> {bindc_name = ".tmp.array", uniq_name = ""}
  %7 = fir.declare %6(%1) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<100000xi32>>, !fir.shape<1>) -> !fir.heap<!fir.array<100000xi32>>
  fir.do_loop %arg1 = %c1 to %c100000 step %c1 unordered {
    %9 = fir.array_coor %2(%1) %arg1 : (!fir.ref<!fir.array<100000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    %10 = fir.load %9 : !fir.ref<f32>
    %11 = fir.convert %10 : (f32) -> i32
    %12 = fir.array_coor %7(%1) %arg1 : (!fir.heap<!fir.array<100000xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    fir.store %11 to %12 : !fir.ref<i32>
  }
  %8 = fir.convert %7 : (!fir.heap<!fir.array<100000xi32>>) -> !fir.ref<!fir.array<100000xi32>>
  fir.call @_QPibar(%8) fastmath<contract> : (!fir.ref<!fir.array<100000xi32>>) -> ()
  fir.freemem %7 : !fir.heap<!fir.array<100000xi32>>
  return
}
// CHECK-LABEL:   func.func @_QPcst_alloca(
// CHECK-DAG:           %[[VAL_0:.*]] = fir.alloca !fir.array<100000xf32> {bindc_name = ".tmp.array", fir.has_lifetime}
// CHECK-DAG:           %[[VAL_2:.*]] = fir.alloca !fir.array<100000xi32> {bindc_name = ".tmp.array", fir.has_lifetime}
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<100000xf32>>) -> !llvm.ptr
// CHECK:           llvm.intr.lifetime.start %[[VAL_9]] : !llvm.ptr
// CHECK:           fir.do_loop
// CHECK:           fir.call @_QPbar(
// CHECK:           llvm.intr.lifetime.end %[[VAL_9]] : !llvm.ptr
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<100000xi32>>) -> !llvm.ptr
// CHECK:           llvm.intr.lifetime.start %[[VAL_17]] : !llvm.ptr
// CHECK:           fir.do_loop
// CHECK:           fir.call @_QPibar(
// CHECK:           llvm.intr.lifetime.end %[[VAL_17]] : !llvm.ptr


func.func @_QPdyn_alloca(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i64> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFdyn_allocaEn"} : (!fir.ref<i64>, !fir.dscope) -> !fir.ref<i64>
  %2 = fir.load %1 : !fir.ref<i64>
  %3 = fir.convert %2 : (i64) -> index
  %4 = arith.cmpi sgt, %3, %c0 : index
  %5 = arith.select %4, %3, %c0 : index
  %6 = fir.shape %5 : (index) -> !fir.shape<1>
  %7 = fir.declare %arg0(%6) dummy_scope %0 {uniq_name = "_QFdyn_allocaEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xf32>>
  %8 = fir.allocmem !fir.array<?xf32>, %5 {bindc_name = ".tmp.array", uniq_name = ""}
  %9 = fir.declare %8(%6) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.heap<!fir.array<?xf32>>
  fir.do_loop %arg2 = %c1 to %5 step %c1 unordered {
    %14 = fir.array_coor %7(%6) %arg2 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    %15 = fir.load %14 : !fir.ref<f32>
    %16 = arith.addf %15, %15 fastmath<contract> : f32
    %17 = fir.array_coor %9(%6) %arg2 : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    fir.store %16 to %17 : !fir.ref<f32>
  }
  %10 = fir.convert %9 : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  fir.call @_QPbar(%10) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>) -> ()
  fir.freemem %9 : !fir.heap<!fir.array<?xf32>>
  %11 = fir.allocmem !fir.array<?xf32>, %5 {bindc_name = ".tmp.array", uniq_name = ""}
  %12 = fir.declare %11(%6) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.heap<!fir.array<?xf32>>
  fir.do_loop %arg2 = %c1 to %5 step %c1 unordered {
    %14 = fir.array_coor %7(%6) %arg2 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    %15 = fir.load %14 : !fir.ref<f32>
    %16 = arith.mulf %15, %15 fastmath<contract> : f32
    %17 = fir.array_coor %12(%6) %arg2 : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    fir.store %16 to %17 : !fir.ref<f32>
  }
  %13 = fir.convert %12 : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  fir.call @_QPbar(%13) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>) -> ()
  fir.freemem %12 : !fir.heap<!fir.array<?xf32>>
  return
}
// CHECK-LABEL:   func.func @_QPdyn_alloca(
// CHECK-NOT:       llvm.intr.lifetime.start
// CHECK:           return

func.func private @_QPbar(!fir.ref<!fir.array<100000xf32>>)
func.func private @_QPibar(!fir.ref<!fir.array<100000xi32>>)
}
