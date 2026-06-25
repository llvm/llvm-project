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

// CHECK: define [[FNATTRS]] <2 x i1> @
// CHECK: %hlsl.isinf = call <2 x i1> @llvm.[[TARGET]].isinf.v2f32
// CHECK: ret <2 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool1x2 test_isinf_double1x2(double1x2 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <3 x i1> @
// CHECK: %hlsl.isinf = call <3 x i1> @llvm.[[TARGET]].isinf.v3f32
// CHECK: ret <3 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool1x3 test_isinf_double1x3(double1x3 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <4 x i1> @
// CHECK: %hlsl.isinf = call <4 x i1> @llvm.[[TARGET]].isinf.v4f32
// CHECK: ret <4 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool1x4 test_isinf_double1x4(double1x4 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <2 x i1> @
// CHECK: %hlsl.isinf = call <2 x i1> @llvm.[[TARGET]].isinf.v2f32
// CHECK: ret <2 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool2x1 test_isinf_double2x1(double2x1 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <4 x i1> @
// CHECK: %hlsl.isinf = call <4 x i1> @llvm.[[TARGET]].isinf.v4f32
// CHECK: ret <4 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool2x2 test_isinf_double2x2(double2x2 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <6 x i1> @
// CHECK: %hlsl.isinf = call <6 x i1> @llvm.[[TARGET]].isinf.v6f32
// CHECK: ret <6 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool2x3 test_isinf_double2x3(double2x3 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <8 x i1> @
// CHECK: %hlsl.isinf = call <8 x i1> @llvm.[[TARGET]].isinf.v8f32
// CHECK: ret <8 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool2x4 test_isinf_double2x4(double2x4 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <3 x i1> @
// CHECK: %hlsl.isinf = call <3 x i1> @llvm.[[TARGET]].isinf.v3f32
// CHECK: ret <3 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool3x1 test_isinf_double3x1(double3x1 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <6 x i1> @
// CHECK: %hlsl.isinf = call <6 x i1> @llvm.[[TARGET]].isinf.v6f32
// CHECK: ret <6 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool3x2 test_isinf_double3x2(double3x2 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <9 x i1> @
// CHECK: %hlsl.isinf = call <9 x i1> @llvm.[[TARGET]].isinf.v9f32
// CHECK: ret <9 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool3x3 test_isinf_double3x3(double3x3 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <12 x i1> @
// CHECK: %hlsl.isinf = call <12 x i1> @llvm.[[TARGET]].isinf.v12f32
// CHECK: ret <12 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool3x4 test_isinf_double3x4(double3x4 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <4 x i1> @
// CHECK: %hlsl.isinf = call <4 x i1> @llvm.[[TARGET]].isinf.v4f32
// CHECK: ret <4 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool4x1 test_isinf_double4x1(double4x1 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <8 x i1> @
// CHECK: %hlsl.isinf = call <8 x i1> @llvm.[[TARGET]].isinf.v8f32
// CHECK: ret <8 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool4x2 test_isinf_double4x2(double4x2 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <12 x i1> @
// CHECK: %hlsl.isinf = call <12 x i1> @llvm.[[TARGET]].isinf.v12f32
// CHECK: ret <12 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool4x3 test_isinf_double4x3(double4x3 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <16 x i1> @
// CHECK: %hlsl.isinf = call <16 x i1> @llvm.[[TARGET]].isinf.v16f32
// CHECK: ret <16 x i1> %hlsl.isinf
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool4x4 test_isinf_double4x4(double4x4 p0) { return isinf(p0); }
