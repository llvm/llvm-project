// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s -DTARGET=dx
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s -DTARGET=spv
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -verify -verify-ignore-unexpected=note
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple spirv-unknown-vulkan-library %s \
// RUN:   -verify -verify-ignore-unexpected=note

// CHECK-LABEL: define {{.*}} <2 x i1> @_{{.*}}test_isnan_double1x2{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[RET:%.*]] = call <2 x i1> @llvm.[[TARGET]].isnan.v2f32(<2 x float> [[CONV]])
// CHECK:    ret <2 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool1x2 test_isnan_double1x2(double1x2 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <3 x i1> @_{{.*}}test_isnan_double1x3{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[RET:%.*]] = call <3 x i1> @llvm.[[TARGET]].isnan.v3f32(<3 x float> [[CONV]])
// CHECK:    ret <3 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool1x3 test_isnan_double1x3(double1x3 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <4 x i1> @_{{.*}}test_isnan_double1x4{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[RET:%.*]] = call <4 x i1> @llvm.[[TARGET]].isnan.v4f32(<4 x float> [[CONV]])
// CHECK:    ret <4 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool1x4 test_isnan_double1x4(double1x4 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <2 x i1> @_{{.*}}test_isnan_double2x1{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[RET:%.*]] = call <2 x i1> @llvm.[[TARGET]].isnan.v2f32(<2 x float> [[CONV]])
// CHECK:    ret <2 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool2x1 test_isnan_double2x1(double2x1 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <4 x i1> @_{{.*}}test_isnan_double2x2{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[RET:%.*]] = call <4 x i1> @llvm.[[TARGET]].isnan.v4f32(<4 x float> [[CONV]])
// CHECK:    ret <4 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool2x2 test_isnan_double2x2(double2x2 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <6 x i1> @_{{.*}}test_isnan_double2x3{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <6 x double> %{{.*}} to <6 x float>
// CHECK:    [[RET:%.*]] = call <6 x i1> @llvm.[[TARGET]].isnan.v6f32(<6 x float> [[CONV]])
// CHECK:    ret <6 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool2x3 test_isnan_double2x3(double2x3 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <8 x i1> @_{{.*}}test_isnan_double2x4{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <8 x double> %{{.*}} to <8 x float>
// CHECK:    [[RET:%.*]] = call <8 x i1> @llvm.[[TARGET]].isnan.v8f32(<8 x float> [[CONV]])
// CHECK:    ret <8 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool2x4 test_isnan_double2x4(double2x4 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <3 x i1> @_{{.*}}test_isnan_double3x1{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[RET:%.*]] = call <3 x i1> @llvm.[[TARGET]].isnan.v3f32(<3 x float> [[CONV]])
// CHECK:    ret <3 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool3x1 test_isnan_double3x1(double3x1 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <6 x i1> @_{{.*}}test_isnan_double3x2{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <6 x double> %{{.*}} to <6 x float>
// CHECK:    [[RET:%.*]] = call <6 x i1> @llvm.[[TARGET]].isnan.v6f32(<6 x float> [[CONV]])
// CHECK:    ret <6 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool3x2 test_isnan_double3x2(double3x2 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <9 x i1> @_{{.*}}test_isnan_double3x3{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <9 x double> %{{.*}} to <9 x float>
// CHECK:    [[RET:%.*]] = call <9 x i1> @llvm.[[TARGET]].isnan.v9f32(<9 x float> [[CONV]])
// CHECK:    ret <9 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool3x3 test_isnan_double3x3(double3x3 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <12 x i1> @_{{.*}}test_isnan_double3x4{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <12 x double> %{{.*}} to <12 x float>
// CHECK:    [[RET:%.*]] = call <12 x i1> @llvm.[[TARGET]].isnan.v12f32(<12 x float> [[CONV]])
// CHECK:    ret <12 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool3x4 test_isnan_double3x4(double3x4 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <4 x i1> @_{{.*}}test_isnan_double4x1{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[RET:%.*]] = call <4 x i1> @llvm.[[TARGET]].isnan.v4f32(<4 x float> [[CONV]])
// CHECK:    ret <4 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool4x1 test_isnan_double4x1(double4x1 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <8 x i1> @_{{.*}}test_isnan_double4x2{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <8 x double> %{{.*}} to <8 x float>
// CHECK:    [[RET:%.*]] = call <8 x i1> @llvm.[[TARGET]].isnan.v8f32(<8 x float> [[CONV]])
// CHECK:    ret <8 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool4x2 test_isnan_double4x2(double4x2 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <12 x i1> @_{{.*}}test_isnan_double4x3{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <12 x double> %{{.*}} to <12 x float>
// CHECK:    [[RET:%.*]] = call <12 x i1> @llvm.[[TARGET]].isnan.v12f32(<12 x float> [[CONV]])
// CHECK:    ret <12 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool4x3 test_isnan_double4x3(double4x3 p0) { return isnan(p0); }

// CHECK-LABEL: define {{.*}} <16 x i1> @_{{.*}}test_isnan_double4x4{{.*}}(
// CHECK:    [[CONV:%.*]] = fptrunc {{.*}} <16 x double> %{{.*}} to <16 x float>
// CHECK:    [[RET:%.*]] = call <16 x i1> @llvm.[[TARGET]].isnan.v16f32(<16 x float> [[CONV]])
// CHECK:    ret <16 x i1> [[RET]]
// expected-warning@+1 {{'isnan' is deprecated: In 202x 64 bit API lowering for isnan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
bool4x4 test_isnan_double4x4(double4x4 p0) { return isnan(p0); }
