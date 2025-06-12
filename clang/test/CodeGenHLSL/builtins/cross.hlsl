// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DFNATTRS="noundef nofpclass(nan inf)" -DTARGET=dx
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DFNATTRS="noundef nofpclass(nan inf)" -DTARGET=dx
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DFNATTRS="spir_func noundef nofpclass(nan inf)" -DTARGET=spv
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DFNATTRS="spir_func noundef nofpclass(nan inf)" -DTARGET=spv

// NATIVE_HALF: define [[FNATTRS]] <3 x half> @
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.[[TARGET]].cross.v3f16(<3 x half>
// NATIVE_HALF: ret <3 x half> %hlsl.cross
// NO_HALF: define [[FNATTRS]] <3 x float> @
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].cross.v3f32(<3 x float>
// NO_HALF: ret <3 x float> %hlsl.cross
half3 test_cross_half3(half3 p0, half3 p1)
{
    return cross(p0, p1);
}

// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %hlsl.cross = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].cross.v3f32(
// CHECK: ret <3 x float> %hlsl.cross
float3 test_cross_float3(float3 p0, float3 p1)
{
    return cross(p0, p1);
}
