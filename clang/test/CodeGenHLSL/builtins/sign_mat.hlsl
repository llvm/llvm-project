// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DTARGET=dx -DFNATTRS="hidden noundef"
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DTARGET=dx -DFNATTRS="hidden noundef"
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DTARGET=spv -DFNATTRS="hidden spir_func noundef"
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DTARGET=spv -DFNATTRS="hidden spir_func noundef"

// CHECK-LABEL: test_sign_half2x3
// NATIVE_HALF: call <6 x i32> @llvm.[[TARGET]].sign.v6f16(<6 x half> %{{.*}})
// NO_HALF: call <6 x i32> @llvm.[[TARGET]].sign.v6f32(<6 x float> %{{.*}})
int2x3 test_sign_half2x3(half2x3 p0) { return sign(p0); }

// CHECK-LABEL: test_sign_half3x4
// NATIVE_HALF: call <12 x i32> @llvm.[[TARGET]].sign.v12f16(<12 x half> %{{.*}})
// NO_HALF: call <12 x i32> @llvm.[[TARGET]].sign.v12f32(<12 x float> %{{.*}})
int3x4 test_sign_half3x4(half3x4 p0) { return sign(p0); }

// CHECK-LABEL: test_sign_half4x4
// NATIVE_HALF: call <16 x i32> @llvm.[[TARGET]].sign.v16f16(<16 x half> %{{.*}})
// NO_HALF: call <16 x i32> @llvm.[[TARGET]].sign.v16f32(<16 x float> %{{.*}})
int4x4 test_sign_half4x4(half4x4 p0) { return sign(p0); }

// CHECK-LABEL: test_sign_int2x3
// CHECK: call <6 x i32> @llvm.[[TARGET]].sign.v6i32(<6 x i32> %{{.*}})
int2x3 test_sign_int2x3(int2x3 p0) { return sign(p0); }

// CHECK-LABEL: test_sign_int3x4
// CHECK: call <12 x i32> @llvm.[[TARGET]].sign.v12i32(<12 x i32> %{{.*}})
int3x4 test_sign_int3x4(int3x4 p0) { return sign(p0); }

// CHECK-LABEL: test_sign_int4x4
// CHECK: call <16 x i32> @llvm.[[TARGET]].sign.v16i32(<16 x i32> %{{.*}})
int4x4 test_sign_int4x4(int4x4 p0) { return sign(p0); }

// CHECK-LABEL: test_sign_uint2x3
// CHECK: [[CMP:%.*]] = icmp eq <6 x i32> %{{.*}}, zeroinitializer
// CHECK: %hlsl.sign = select <6 x i1> [[CMP]], <6 x i32> zeroinitializer, <6 x i32> splat (i32 1)
int2x3 test_sign_uint2x3(uint2x3 p0) { return sign(p0); }

// CHECK-LABEL: test_sign_uint3x4
// CHECK: [[CMP:%.*]] = icmp eq <12 x i32> %{{.*}}, zeroinitializer
// CHECK: %hlsl.sign = select <12 x i1> [[CMP]], <12 x i32> zeroinitializer, <12 x i32> splat (i32 1)
int3x4 test_sign_uint3x4(uint3x4 p0) { return sign(p0); }

// CHECK-LABEL: test_sign_uint4x4
// CHECK: [[CMP:%.*]] = icmp eq <16 x i32> %{{.*}}, zeroinitializer
// CHECK: %hlsl.sign = select <16 x i1> [[CMP]], <16 x i32> zeroinitializer, <16 x i32> splat (i32 1)
int4x4 test_sign_uint4x4(uint4x4 p0) { return sign(p0); }
