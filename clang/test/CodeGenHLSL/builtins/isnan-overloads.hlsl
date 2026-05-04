// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s \
// RUN:   -DFNATTRS="hidden noundef" -DTARGET=dx
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s \
// RUN:   -DFNATTRS="hidden spir_func noundef" -DTARGET=spv
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -verify -verify-ignore-unexpected=note
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple spirv-unknown-vulkan-compute %s \
// RUN:   -verify -verify-ignore-unexpected=note

// CHECK: define [[FNATTRS]] i1 @
// CHECK: %hlsl.isnan = call i1 @llvm.[[TARGET]].isnan.f32(
// CHECK: ret i1 %hlsl.isnan
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool test_isnan_double(double p0) { return isnan(p0); }
// CHECK: define [[FNATTRS]] <2 x i1> @
// CHECK: %hlsl.isnan = call <2 x i1> @llvm.[[TARGET]].isnan.v2f32
// CHECK: ret <2 x i1> %hlsl.isnan
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool2 test_isnan_double2(double2 p0) { return isnan(p0); }
// CHECK: define [[FNATTRS]] <3 x i1> @
// CHECK: %hlsl.isnan = call <3 x i1> @llvm.[[TARGET]].isnan.v3f32
// CHECK: ret <3 x i1> %hlsl.isnan
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool3 test_isnan_double3(double3 p0) { return isnan(p0); }
// CHECK: define [[FNATTRS]] <4 x i1> @
// CHECK: %hlsl.isnan = call <4 x i1> @llvm.[[TARGET]].isnan.v4f32
// CHECK: ret <4 x i1> %hlsl.isnan
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool4 test_isnan_double4(double4 p0) { return isnan(p0); }
