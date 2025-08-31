// RUN: tco -emit-final-mlir --force-no-alias=false %s | FileCheck %s --check-prefixes=CHECK,AA,CMPLX
// RUN: tco -emit-final-mlir --force-no-alias=false -enable-aa=false %s | FileCheck %s --check-prefixes=CHECK,NOAA,CMPLX
// RUN: tco -emit-final-mlir --force-no-alias=false -simplify-mlir %s | FileCheck %s --check-prefixes=CHECK,AA,SIMPLE
// RUN: tco -emit-final-mlir --force-no-alias=false -enable-aa=false -simplify-mlir %s | FileCheck %s --check-prefixes=CHECK,NOAA,SIMPLE
// RUN: tco -test-gen --force-no-alias=false %s | FileCheck %s --check-prefixes=CHECK,NOAA,SIMPLE

// Just a dummy function that exhibits all of the things we want to turn on and off
func.func @_QPtest(%arg0: !fir.ref<i32> {fir.bindc_name = "num"}, %arg1: !fir.ref<i32> {fir.bindc_name = "lb"}, %arg2: !fir.ref<i32> {fir.bindc_name = "ub"}, %arg3: !fir.ref<i32> {fir.bindc_name = "step"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtestEi"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFtestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtestElb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtestEnum"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg3 dummy_scope %0 {uniq_name = "_QFtestEstep"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFtestEub"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.load %3#0 : !fir.ref<i32>
  %8 = fir.convert %7 : (i32) -> index
  %9 = fir.load %6#0 : !fir.ref<i32>
  %10 = fir.convert %9 : (i32) -> index
  %11 = fir.load %5#0 : !fir.ref<i32>
  %12 = fir.convert %11 : (i32) -> index
  %13 = fir.convert %8 : (index) -> i32
  %14:2 = fir.do_loop %arg4 = %8 to %10 step %12 iter_args(%arg5 = %13) -> (index, i32) {
    fir.store %arg5 to %2#0 : !fir.ref<i32>
    %15 = fir.load %4#0 : !fir.ref<i32>
    %16 = fir.load %2#0 : !fir.ref<i32>
    %17 = arith.addi %15, %16 : i32
    hlfir.assign %17 to %4#0 : i32, !fir.ref<i32>
    %18 = arith.addi %arg4, %12 overflow<nsw> : index
    %19 = fir.convert %12 : (index) -> i32
    %20 = fir.load %2#0 : !fir.ref<i32>
    %21 = arith.addi %20, %19 overflow<nsw> : i32
    fir.result %18, %21 : index, i32
  }
  fir.store %14#1 to %2#0 : !fir.ref<i32>
  return
}

// CHECK-LABEL:   llvm.func @_QPtest(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr {fir.bindc_name = "num", llvm.nocapture},
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.ptr {fir.bindc_name = "lb", llvm.nocapture},
// CHECK-SAME:      %[[ARG2:.*]]: !llvm.ptr {fir.bindc_name = "ub", llvm.nocapture},
// CHECK-SAME:      %[[ARG3:.*]]: !llvm.ptr {fir.bindc_name = "step", llvm.nocapture}) {

// CMPLX:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CMPLX:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
// CMPLX:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : index) : i64
// CMPLX:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : index) : i64
// CMPLX:           %[[VAL_4:.*]] = llvm.mlir.constant(1 : i64) : i64

// SIMPLE:          %[[VAL_3:.*]] = llvm.mlir.constant(0 : index) : i64
// SIMPLE:          %[[VAL_2:.*]] = llvm.mlir.constant(1 : index) : i64
// SIMPLE:          %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// SIMPLE:          %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr

// AA:              %[[VAL_5:.*]] = llvm.load %[[ARG1]] {tbaa = [#llvm.tbaa_tag<base_type = <id = "dummy arg data/_QFtestElb", members = {<#llvm.tbaa_type_desc<id = "dummy arg data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, access_type = <id = "dummy arg data/_QFtestElb", members = {<#llvm.tbaa_type_desc<id = "dummy arg data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
// NOAA:            %[[VAL_5:.*]] = llvm.load %[[ARG1]] : !llvm.ptr -> i32

// CHECK:           %[[VAL_6:.*]] = llvm.sext %[[VAL_5]] : i32 to i64

// AA:              %[[VAL_7:.*]] = llvm.load %[[ARG2]] {tbaa = [#llvm.tbaa_tag<base_type = <id = "dummy arg data/_QFtestEub", members = {<#llvm.tbaa_type_desc<id = "dummy arg data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, access_type = <id = "dummy arg data/_QFtestEub", members = {<#llvm.tbaa_type_desc<id = "dummy arg data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
// NOAA:            %[[VAL_7:.*]] = llvm.load %[[ARG2]] : !llvm.ptr -> i32

// CHECK:           %[[VAL_8:.*]] = llvm.sext %[[VAL_7]] : i32 to i64

// AA:              %[[VAL_9:.*]] = llvm.load %[[ARG3]] {tbaa = [#llvm.tbaa_tag<base_type = <id = "dummy arg data/_QFtestEstep", members = {<#llvm.tbaa_type_desc<id = "dummy arg data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, access_type = <id = "dummy arg data/_QFtestEstep", members = {<#llvm.tbaa_type_desc<id = "dummy arg data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
// NOAA:            %[[VAL_9:.*]] = llvm.load %[[ARG3]] : !llvm.ptr -> i32

// CHECK:           %[[VAL_10:.*]] = llvm.sext %[[VAL_9]] : i32 to i64
// CHECK:           %[[VAL_11:.*]] = llvm.trunc %[[VAL_6]] : i64 to i32
// CHECK:           %[[VAL_12:.*]] = llvm.sub %[[VAL_8]], %[[VAL_6]] : i64
// CHECK:           %[[VAL_13:.*]] = llvm.add %[[VAL_12]], %[[VAL_10]] : i64
// CHECK:           %[[VAL_14:.*]] = llvm.sdiv %[[VAL_13]], %[[VAL_10]] : i64
// CHECK:           llvm.br ^bb1(%[[VAL_11]], %[[VAL_14]] : i32, i64)
// CHECK:         ^bb1(%[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: i64):
// CHECK:           %[[VAL_17:.*]] = llvm.icmp "sgt" %[[VAL_16]], %{{.*}} : i64
// CHECK:           llvm.cond_br %[[VAL_17]], ^bb2, ^bb3
// CHECK:         ^bb2:

// AA:              llvm.store %[[VAL_15]], %[[VAL_1]] {tbaa = [#llvm.tbaa_tag<base_type = <id = "allocated data/_QFtestEi", members = {<#llvm.tbaa_type_desc<id = "allocated data", members = {<#llvm.tbaa_type_desc<id = "target data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, 0>}>, access_type = <id = "allocated data/_QFtestEi", members = {<#llvm.tbaa_type_desc<id = "allocated data", members = {<#llvm.tbaa_type_desc<id = "target data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
// NOAA:            llvm.store %[[VAL_15]], %{{.*}} : i32, !llvm.ptr

// AA:              %[[VAL_18:.*]] = llvm.load %[[ARG0]] {tbaa = [#llvm.tbaa_tag<base_type = <id = "dummy arg data/_QFtestEnum", members = {<#llvm.tbaa_type_desc<id = "dummy arg data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, access_type = <id = "dummy arg data/_QFtestEnum", members = {<#llvm.tbaa_type_desc<id = "dummy arg data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
// NOAA:            %[[VAL_18:.*]] = llvm.load %[[ARG0]] : !llvm.ptr -> i32

// AA:              %[[VAL_19:.*]] = llvm.load %[[VAL_1]] {tbaa = [#llvm.tbaa_tag<base_type = <id = "allocated data/_QFtestEi", members = {<#llvm.tbaa_type_desc<id = "allocated data", members = {<#llvm.tbaa_type_desc<id = "target data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, 0>}>, access_type = <id = "allocated data/_QFtestEi", members = {<#llvm.tbaa_type_desc<id = "allocated data", members = {<#llvm.tbaa_type_desc<id = "target data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
// NOAA:            %[[VAL_19:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32

// CHECK:           %[[VAL_20:.*]] = llvm.add %[[VAL_18]], %[[VAL_19]] : i32
// AA:              llvm.store %[[VAL_20]], %[[ARG0]] {tbaa = [#llvm.tbaa_tag<base_type = <id = "dummy arg data/_QFtestEnum", members = {<#llvm.tbaa_type_desc<id = "dummy arg data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, access_type = <id = "dummy arg data/_QFtestEnum", members = {<#llvm.tbaa_type_desc<id = "dummy arg data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
// NOAA:            llvm.store %[[VAL_20]], %[[ARG0]] : i32, !llvm.ptr

// CHECK:           %[[VAL_21:.*]] = llvm.trunc %[[VAL_10]] : i64 to i32

// AA:              %[[VAL_22:.*]] = llvm.load %[[VAL_1]] {tbaa = [#llvm.tbaa_tag<base_type = <id = "allocated data/_QFtestEi", members = {<#llvm.tbaa_type_desc<id = "allocated data", members = {<#llvm.tbaa_type_desc<id = "target data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, 0>}>, access_type = <id = "allocated data/_QFtestEi", members = {<#llvm.tbaa_type_desc<id = "allocated data", members = {<#llvm.tbaa_type_desc<id = "target data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
// NOAA:            %[[VAL_22:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32

// CHECK:           %[[VAL_23:.*]] = llvm.add %[[VAL_22]], %[[VAL_21]] overflow<nsw> : i32
// CHECK:           %[[VAL_24:.*]] = llvm.sub %[[VAL_16]], %{{.*}} : i64
// CHECK:           llvm.br ^bb1(%[[VAL_23]], %[[VAL_24]] : i32, i64)
// CHECK:         ^bb3:

// AA:              llvm.store %[[VAL_15]], %[[VAL_1]] {tbaa = [#llvm.tbaa_tag<base_type = <id = "allocated data/_QFtestEi", members = {<#llvm.tbaa_type_desc<id = "allocated data", members = {<#llvm.tbaa_type_desc<id = "target data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, 0>}>, access_type = <id = "allocated data/_QFtestEi", members = {<#llvm.tbaa_type_desc<id = "allocated data", members = {<#llvm.tbaa_type_desc<id = "target data", members = {<#llvm.tbaa_type_desc<id = "any data access", members = {<#llvm.tbaa_type_desc<id = "any access", members = {<#llvm.tbaa_root<id = "Flang function root _QPtest">, 0>}>, 0>}>, 0>}>, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
// NOAA:            llvm.store %[[VAL_15]], %{{.*}} : i32, !llvm.ptr

// CHECK:           llvm.return
// CHECK:         }

