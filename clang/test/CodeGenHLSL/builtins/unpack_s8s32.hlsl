// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s -check-prefix=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s -check-prefix=CHECK,CHECK-SPV

// CHECK: define {{.*}} <4 x i32> @_Z10test_s8s32u14int8_t4_packed
// CHECK-DXIL: [[VAR:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.unpack.s8s32(i32 %{{.*}})
// CHECK-DXIL: %{{.*}} = extractvalue { i32, i32, i32, i32 } [[VAR]], 0
// CHECK-DXIL: %{{.*}} = insertelement <4 x i32>
// CHECK-DXIL: %{{.*}} = extractvalue { i32, i32, i32, i32 } [[VAR]], 1
// CHECK-DXIL: %{{.*}} = insertelement <4 x i32>
// CHECK-DXIL: %{{.*}} = extractvalue { i32, i32, i32, i32 } [[VAR]], 2
// CHECK-DXIL: %{{.*}} = insertelement <4 x i32>
// CHECK-DXIL: %{{.*}} = extractvalue { i32, i32, i32, i32 } [[VAR]], 3
// CHECK-DXIL: %{{.*}} = insertelement <4 x i32>
// CHECK-DXIL: ret <4 x i32>
// CHECK-SPV: [[VAR:%.*]] = call <4 x i32> @llvm.spv.unpack.s8s32(i32 %{{.*}})
// CHECK-SPV: ret <4 x i32> [[VAR]]
int32_t4 test_s8s32(int8_t4_packed val) { return unpack_s8s32(val); }
