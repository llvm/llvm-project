// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s -DTARGET=dx
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s -DTARGET=spv
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -verify \
// RUN:   -verify-ignore-unexpected=note
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -verify \
// RUN:   -verify-ignore-unexpected=note

// CHECK: define {{.*}} i1 @_Z17test_isnan_doubled(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[RET:%.*]] = call noundef i1 @llvm.[[TARGET]].isnan.f32(float [[CONV]])
// CHECK:    ret i1 [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool test_isnan_double(double p0) { return isnan(p0); }
// CHECK: define {{.*}} <2 x i1> @_Z18test_isnan_double2Dv2_d(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[RET:%.*]] = call noundef <2 x i1> @llvm.[[TARGET]].isnan.v2f32(<2 x float> [[CONV]])
// CHECK:    ret <2 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool2 test_isnan_double2(double2 p0) { return isnan(p0); }
// CHECK: define {{.*}} <3 x i1> @_Z18test_isnan_double3Dv3_d(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[RET:%.*]] = call noundef <3 x i1> @llvm.[[TARGET]].isnan.v3f32(<3 x float> [[CONV]])
// CHECK:    ret <3 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool3 test_isnan_double3(double3 p0) { return isnan(p0); }
// CHECK: define {{.*}} <4 x i1> @_Z18test_isnan_double4Dv4_d(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[RET:%.*]] = call noundef <4 x i1> @llvm.[[TARGET]].isnan.v4f32(<4 x float> [[CONV]])
// CHECK:    ret <4 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool4 test_isnan_double4(double4 p0) { return isnan(p0); }
