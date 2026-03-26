// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute %s \
// RUN:  -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s

 half2 half_vec_mod_by_int(half2 p1) {
// CHECK-LABEL: half_vec_mod_by_int
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn <2 x half> %{{.*}}, splat (half 0xH4000)
    return  p1 % 2;
}

 half2 half_vec_mod_by_float(half2 p1) {
// CHECK-LABEL: half_vec_mod_by_float
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn <2 x half> %{{.*}}, splat (half 0xH4000)
    return  p1 % (half)2.0;
}

 half2 half_vec_mod_by_half(half2 p1, half p2 ) {
// CHECK-LABEL: half_vec_mod_by_half
// CHECK:  %splat.splatinsert = insertelement <2 x half> poison, half %{{.*}}, i64 0
// CHECK:  %splat.splat = shufflevector <2 x half> %splat.splatinsert, <2 x half> poison, <2 x i32> zeroinitializer
// CHECK:  %rem = frem reassoc nnan ninf nsz arcp afn <2 x half> %{{.*}}, %splat.splat
    return  p1 % p2;
}

 half2 half_vec_mod_by_half_vec(half2 p1, half2 p2 ) {
// CHECK-LABEL: half_vec_mod_by_half_vec
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn <2 x half> %{{.*}}, %{{.*}}
    return  p1 % p2;
}

 half half_vec_mod_by_int(half p1) {
// CHECK-LABEL: half_vec_mod_by_int
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn half  %{{.*}}, 0xH4000
    return  p1 % 2;
}

 half half_mod_by_float(half p1) {
// CHECK-LABEL: half_mod_by_float
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn half  %{{.*}}, 0xH4000
    return  p1 % (half)2.0;
}

 half half_mod_by_half(half p1, half p2 ) {
// CHECK-LABEL: half_mod_by_half
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn half %{{.*}}, %{{.*}}
    return  p1 % p2;
}

 half half_mod_by_half_vec(half p1, half2 p2 ) {
// CHECK-LABEL: half_mod_by_half_vec
// CHECK: %splat.splatinsert = insertelement <2 x half> poison, half %{{.*}}, i64 0
// CHECK: %splat.splat = shufflevector <2 x half> %splat.splatinsert, <2 x half> poison, <2 x i32> zeroinitializer
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn <2 x half> %splat.splat, %{{.*}}
    return  p1 % p2;
}

 float2 float_vec_mod_by_int(float2 p1) {
// CHECK-LABEL: float_vec_mod_by_int
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, splat (float 2.000000e+00)
    return  p1 % 2;
}

 float2 float_vec_mod_by_float_const(float2 p1) {
// CHECK-LABEL: float_vec_mod_by_float_const
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, splat (float 2.000000e+00)
    return  p1 % 2.0;
}

 float2 float_vec_mod_by_float(float2 p1, float p2 ) {
// CHECK-LABEL: float_vec_mod_by_float
// CHECK:  %splat.splatinsert = insertelement <2 x float> poison, float %{{.*}}, i64 0
// CHECK:  %splat.splat = shufflevector <2 x float> %splat.splatinsert, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK:  %rem = frem reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, %splat.splat
    return  p1 % p2;
}

 float2 float_vec_mod_by_float_vec(float2 p1, float2 p2 ) {
// CHECK-LABEL: float_vec_mod_by_float_vec
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn <2 x float> %{{.*}}, %{{.*}}
    return  p1 % p2;
}

 float float_mod_by_int(float p1) {
// CHECK-LABEL: float_mod_by_int
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn float %{{.*}}, 2.000000e+00
    return  p1 % 2;
}

 float float_mod_by_float_const(float p1) {
// CHECK-LABEL: float_mod_by_float_const
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn float %{{.*}}, 2.000000e+00
    return  p1 % 2.0;
}

 float float_mod_by_float(float p1, float p2 ) {
// CHECK-LABEL: float_mod_by_float
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
    return  p1 % p2;
}

 float float_mod_by_float_vec(float p1, float2 p2 ) {
// CHECK-LABEL: float_mod_by_float_vec
// CHECK: %splat.splatinsert = insertelement <2 x float> poison, float %{{.*}}, i64 0
// CHECK: %splat.splat = shufflevector <2 x float> %splat.splatinsert, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK: %rem = frem reassoc nnan ninf nsz arcp afn <2 x float> %splat.splat, %{{.*}}
    return  p1 % p2;
}
