// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:   -DFNATTRS="hidden noundef" -DTARGET=dx
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:   -DFNATTRS="hidden spir_func noundef" -DTARGET=spv
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s  \
// RUN:   -verify -verify-ignore-unexpected=note
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple spirv-unknown-vulkan-compute %s  \
// RUN:   -verify -verify-ignore-unexpected=note

// CHECK: define [[FNATTRS]] i1 @_Z17test_isinf_doubled(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[HLSLISINFI:%.*]] = call noundef i1 @llvm.[[TARGET]].isinf.f32(float [[CONVI]])
// CHECK:    ret i1 [[HLSLISINFI]]
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool test_isinf_double(double p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <2 x i1> @_Z18test_isinf_double2Dv2_d(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[HLSLISINFI:%.*]] = call noundef <2 x i1> @llvm.[[TARGET]].isinf.v2f32(<2 x float> [[CONVI]])
// CHECK:    ret <2 x i1> [[HLSLISINFI]]
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool2 test_isinf_double2(double2 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <3 x i1> @_Z18test_isinf_double3Dv3_d(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[HLSLISINFI:%.*]] = call noundef <3 x i1> @llvm.[[TARGET]].isinf.v3f32(<3 x float> [[CONVI]])
// CHECK:    ret <3 x i1> [[HLSLISINFI]]
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool3 test_isinf_double3(double3 p0) { return isinf(p0); }
// CHECK: define [[FNATTRS]] <4 x i1> @_Z18test_isinf_double4Dv4_d(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[HLSLISINFI:%.*]] = call noundef <4 x i1> @llvm.[[TARGET]].isinf.v4f32(<4 x float> [[CONVI]])
// CHECK:    ret <4 x i1> [[HLSLISINFI]]
// expected-warning@+1 {{'isinf' is deprecated: In 202x 64 bit API lowering for fn is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool4 test_isinf_double4(double4 p0) { return isinf(p0); }
