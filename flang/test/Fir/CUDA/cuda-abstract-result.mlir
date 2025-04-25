// RUN: fir-opt -pass-pipeline='builtin.module(gpu.module(gpu.func(abstract-result)))' %s | FileCheck %s

gpu.module @test {
 gpu.func @_QMinterval_mPtest1(%arg0: !fir.ref<!fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>>, %arg1: !fir.ref<f32>) -> !fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}> {
    %c1_i32 = arith.constant 1 : i32
    %18 = fir.dummy_scope : !fir.dscope
    %19 = fir.declare %arg0 dummy_scope %18 {uniq_name = "_QMinterval_mFtest1Ea"} : (!fir.ref<!fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>>, !fir.dscope) -> !fir.ref<!fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>>
    %20 = fir.declare %arg1 dummy_scope %18 {uniq_name = "_QMinterval_mFtest1Eb"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
    %21 = fir.alloca !fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}> {bindc_name = "c", uniq_name = "_QMinterval_mFtest1Ec"}
    %22 = fir.declare %21 {uniq_name = "_QMinterval_mFtest1Ec"} : (!fir.ref<!fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>>) -> !fir.ref<!fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>>
    %23 = fir.alloca i32 {bindc_name = "warpsize", uniq_name = "_QMcudadeviceECwarpsize"}
    %24 = fir.declare %23 {uniq_name = "_QMcudadeviceECwarpsize"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %25 = fir.field_index inf, !fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>
    %26 = fir.coordinate_of %19, %25 : (!fir.ref<!fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>>, !fir.field) -> !fir.ref<f32>
    %27 = fir.load %20 : !fir.ref<f32>
    %28 = arith.negf %27 fastmath<contract> : f32
    %29 = fir.load %26 : !fir.ref<f32>
    %30 = fir.call @__fadd_rd(%29, %28) proc_attrs<bind_c> fastmath<contract> : (f32, f32) -> f32
    %31 = fir.field_index inf, !fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>
    %32 = fir.coordinate_of %22, %31 : (!fir.ref<!fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>>, !fir.field) -> !fir.ref<f32>
    fir.store %30 to %32 : !fir.ref<f32>
    %33 = fir.field_index sup, !fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>
    %34 = fir.coordinate_of %19, %33 : (!fir.ref<!fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>>, !fir.field) -> !fir.ref<f32>
    %35 = fir.load %20 : !fir.ref<f32>
    %36 = arith.negf %35 fastmath<contract> : f32
    %37 = fir.load %34 : !fir.ref<f32>
    %38 = fir.call @__fadd_ru(%37, %36) proc_attrs<bind_c> fastmath<contract> : (f32, f32) -> f32
    %39 = fir.field_index sup, !fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>
    %40 = fir.coordinate_of %22, %39 : (!fir.ref<!fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>>, !fir.field) -> !fir.ref<f32>
    fir.store %38 to %40 : !fir.ref<f32>
    %41 = fir.load %22 : !fir.ref<!fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>>
    gpu.return %41 : !fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>
  }
}

// CHECK: gpu.func @_QMinterval_mPtest1(%arg0: !fir.ref<!fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>>, %arg1: !fir.ref<!fir.type<_QMinterval_mTinterval{inf:f32,sup:f32}>>, %arg2: !fir.ref<f32>) {
// CHECK: gpu.return{{$}}
