// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (git@github.com:SiPearl/llvm-project.git 666e4313ebc03587f27774139ad8f780bac15c3e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
   func.func @_QQmain() attributes {fir.bindc_name = "TEST_CO_SUM"} {
    %0 = fir.dummy_scope : !fir.dscope
    %c2 = arith.constant 2 : index
    %1 = fir.alloca !fir.array<2xf64> {bindc_name = "array_d", uniq_name = "_QFEarray_d"}
    %2 = fir.shape %c2 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %1(%2) {uniq_name = "_QFEarray_d"} : (!fir.ref<!fir.array<2xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xf64>>, !fir.ref<!fir.array<2xf64>>)
    %c2_0 = arith.constant 2 : index
    %4 = fir.alloca !fir.array<2xi32> {bindc_name = "array_i", uniq_name = "_QFEarray_i"}
    %5 = fir.shape %c2_0 : (index) -> !fir.shape<1>
    %6:2 = hlfir.declare %4(%5) {uniq_name = "_QFEarray_i"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
    %c2_1 = arith.constant 2 : index
    %7 = fir.alloca !fir.array<2xf32> {bindc_name = "array_r", uniq_name = "_QFEarray_r"}
    %8 = fir.shape %c2_1 : (index) -> !fir.shape<1>
    %9:2 = hlfir.declare %7(%8) {uniq_name = "_QFEarray_r"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xf32>>, !fir.ref<!fir.array<2xf32>>)
    %10 = fir.alloca f64 {bindc_name = "d", uniq_name = "_QFEd"}
    %11:2 = hlfir.declare %10 {uniq_name = "_QFEd"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %12 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
    %13:2 = hlfir.declare %12 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c1 = arith.constant 1 : index
    %14 = fir.alloca !fir.char<1> {bindc_name = "message", uniq_name = "_QFEmessage"}
    %15:2 = hlfir.declare %14 typeparams %c1 {uniq_name = "_QFEmessage"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    %16 = fir.alloca f32 {bindc_name = "r", uniq_name = "_QFEr"}
    %17:2 = hlfir.declare %16 {uniq_name = "_QFEr"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
    %18 = fir.alloca i32 {bindc_name = "status", uniq_name = "_QFEstatus"}
    %19:2 = hlfir.declare %18 {uniq_name = "_QFEstatus"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %20 = fir.embox %13#0 : (!fir.ref<i32>) -> !fir.box<i32>
    mif.co_sum %20 : (!fir.box<i32>)
    %21 = fir.embox %11#0 : (!fir.ref<f64>) -> !fir.box<f64>
    mif.co_sum %21 : (!fir.box<f64>)
    %22 = fir.embox %17#0 : (!fir.ref<f32>) -> !fir.box<f32>
    mif.co_sum %22 : (!fir.box<f32>)
    %c1_i32 = arith.constant 1 : i32
    %23 = fir.embox %13#0 : (!fir.ref<i32>) -> !fir.box<i32>
    mif.co_sum %23 result %c1_i32 : (!fir.box<i32>, i32)
    %c1_i32_2 = arith.constant 1 : i32
    %24 = fir.embox %11#0 : (!fir.ref<f64>) -> !fir.box<f64>
    %25 = fir.embox %15#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
    mif.co_sum %24 result %c1_i32_2 stat %19#0 errmsg %25 : (!fir.box<f64>, i32, !fir.ref<i32>, !fir.box<!fir.char<1>>)
    %c1_i32_3 = arith.constant 1 : i32
    %26 = fir.embox %17#0 : (!fir.ref<f32>) -> !fir.box<f32>
    %27 = fir.embox %15#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
    mif.co_sum %26 result %c1_i32_3 stat %19#0 errmsg %27 : (!fir.box<f32>, i32, !fir.ref<i32>, !fir.box<!fir.char<1>>)
    %28 = fir.shape %c2_0 : (index) -> !fir.shape<1>
    %29 = fir.embox %6#0(%28) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
    mif.co_sum %29 : (!fir.box<!fir.array<2xi32>>)
    %c1_i32_4 = arith.constant 1 : i32
    %30 = fir.shape %c2 : (index) -> !fir.shape<1>
    %31 = fir.embox %3#0(%30) : (!fir.ref<!fir.array<2xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf64>>
    mif.co_sum %31 result %c1_i32_4 stat %19#0 : (!fir.box<!fir.array<2xf64>>, i32, !fir.ref<i32>)
    %c1_i32_5 = arith.constant 1 : i32
    %32 = fir.shape %c2_1 : (index) -> !fir.shape<1>
    %33 = fir.embox %9#0(%32) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
    %34 = fir.embox %15#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
    mif.co_sum %33 result %c1_i32_5 stat %19#0 errmsg %34 : (!fir.box<!fir.array<2xf32>>, i32, !fir.ref<i32>, !fir.box<!fir.char<1>>)
    return
  }
}

  // CHECK: %[[V1:.*]] = fir.embox %[[VAR_I:.*]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  // CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  // CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[V4:.*]] = fir.absent !fir.ref<i32>
  // CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<i32>) -> !fir.box<none>
  // CHECK: fir.call @_QMprifPprif_co_sum(%[[V5]], %[[V2]], %[[V4]], %[[V3]], %[[V3]]) : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

  // CHECK: %[[V1:.*]] = fir.embox %[[VAR_D:.*]]#0 : (!fir.ref<f64>) -> !fir.box<f64>
  // CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  // CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[V4:.*]] = fir.absent !fir.ref<i32>
  // CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<f64>) -> !fir.box<none>
  // CHECK: fir.call @_QMprifPprif_co_sum(%[[V5]], %[[V2]], %[[V4]], %[[V3]], %[[V3]]) : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

  // CHECK: %[[V1:.*]] = fir.embox %[[VAR_R:.*]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
  // CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  // CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[V4:.*]] = fir.absent !fir.ref<i32>
  // CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<f32>) -> !fir.box<none>
  // CHECK: fir.call @_QMprifPprif_co_sum(%[[V5]], %[[V2]], %[[V4]], %[[V3]], %[[V3]]) : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

  // CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  // CHECK: %[[V1:.*]] = fir.embox %[[VAR_I]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  // CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  // CHECK: %[[V2:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[V3:.*]] = fir.absent !fir.ref<i32>
  // CHECK: %[[V4:.*]] = fir.convert %[[V1]] : (!fir.box<i32>) -> !fir.box<none>
  // CHECK: fir.call @_QMprifPprif_co_sum(%[[V4]], %[[IMAGE_RESULT]], %[[V3]], %[[V2]], %[[V2]]) : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

  // CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  // CHECK: %[[V1:.*]] = fir.embox %[[VAR_D]]#0 : (!fir.ref<f64>) -> !fir.box<f64>
  // CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE:.*]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  // CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  // CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[V4:.*]] = fir.convert %[[V1]] : (!fir.box<f64>) -> !fir.box<none>
  // CHECK: %[[V5:.*]] = fir.convert %[[V2]] : (!fir.box<!fir.char<1>>) -> !fir.box<!fir.char<1,?>>
  // CHECK: fir.call @_QMprifPprif_co_sum(%[[V4]], %[[IMAGE_RESULT]], %[[STATUS:.*]], %[[V5]], %[[V3]]) : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

  // CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  // CHECK: %[[V1:.*]] = fir.embox %[[VAR_R]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
  // CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  // CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  // CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[V4:.*]] = fir.convert %[[V1]] : (!fir.box<f32>) -> !fir.box<none>
  // CHECK: %[[V5:.*]] = fir.convert %[[V2]] : (!fir.box<!fir.char<1>>) -> !fir.box<!fir.char<1,?>>
  // CHECK: fir.call @_QMprifPprif_co_sum(%[[V4]], %[[IMAGE_RESULT]], %[[STATUS]], %[[V5]], %[[V3]]) : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

  // CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  // CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_I:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
  // CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  // CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[V4:.*]] = fir.absent !fir.ref<i32>
  // CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<!fir.array<2xi32>>) -> !fir.box<none>
  // CHECK: fir.call @_QMprifPprif_co_sum(%[[V5]], %[[V2]], %[[V4]], %[[V3]], %[[V3]]) : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
   
  // CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  // CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  // CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_D:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf64>>
  // CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  // CHECK: %[[V2:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[V3:.*]] = fir.convert %[[V1]] : (!fir.box<!fir.array<2xf64>>) -> !fir.box<none>
  // CHECK: fir.call @_QMprifPprif_co_sum(%[[V3]], %[[IMAGE_RESULT]], %[[STATUS]], %[[V2]], %[[V2]]) : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

  // CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  // CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  // CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_C:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
  // CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  // CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  // CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[V4:.*]] = fir.convert %[[V1]] : (!fir.box<!fir.array<2xf32>>) -> !fir.box<none>
  // CHECK: %[[V5:.*]] = fir.convert %[[V2]] : (!fir.box<!fir.char<1>>) -> !fir.box<!fir.char<1,?>>
  // CHECK: fir.call @_QMprifPprif_co_sum(%[[V4]], %[[IMAGE_RESULT]], %[[STATUS]], %[[V5]], %[[V3]]) : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
