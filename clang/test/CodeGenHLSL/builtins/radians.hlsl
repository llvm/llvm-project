// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DTARGET=dx -DFNATTRS="noundef nofpclass(nan inf)"
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DTARGET=dx -DFNATTRS="noundef nofpclass(nan inf)"
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DTARGET=spv -DFNATTRS="spir_func noundef nofpclass(nan inf)"
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DTARGET=spv -DFNATTRS="spir_func noundef nofpclass(nan inf)"


// NATIVE_HALF: define [[FNATTRS]] half @
// NATIVE_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn half @llvm.[[TARGET]].radians.f16(
// NATIVE_HALF: ret half %{{.*}}
// NO_HALF: define [[FNATTRS]] float @
// NO_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].radians.f32(
// NO_HALF: ret float %{{.*}}
half test_radians_half(half p0) { return radians(p0); }
// NATIVE_HALF: define [[FNATTRS]] <2 x half> @
// NATIVE_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.[[TARGET]].radians.v2f16
// NATIVE_HALF: ret <2 x half> %{{.*}}
// NO_HALF: define [[FNATTRS]] <2 x float> @
// NO_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].radians.v2f32(
// NO_HALF: ret <2 x float> %{{.*}}
half2 test_radians_half2(half2 p0) { return radians(p0); }
// NATIVE_HALF: define [[FNATTRS]] <3 x half> @
// NATIVE_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.[[TARGET]].radians.v3f16
// NATIVE_HALF: ret <3 x half> %{{.*}}
// NO_HALF: define [[FNATTRS]] <3 x float> @
// NO_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].radians.v3f32(
// NO_HALF: ret <3 x float> %{{.*}}
half3 test_radians_half3(half3 p0) { return radians(p0); }
// NATIVE_HALF: define [[FNATTRS]] <4 x half> @
// NATIVE_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.[[TARGET]].radians.v4f16
// NATIVE_HALF: ret <4 x half> %{{.*}}
// NO_HALF: define [[FNATTRS]] <4 x float> @
// NO_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].radians.v4f32(
// NO_HALF: ret <4 x float> %{{.*}}
half4 test_radians_half4(half4 p0) { return radians(p0); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].radians.f32(
// CHECK: ret float %{{.*}}
float test_radians_float(float p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].radians.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_radians_float2(float2 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].radians.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_radians_float3(float3 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].radians.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_radians_float4(float4 p0) { return radians(p0); }

