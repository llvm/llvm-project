// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s -check-prefix=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s -check-prefix=CHECK,CHECK-SPV

// CHECK: define {{.*}} <4 x i32> @_Z10test_u8u32
// CHECK-DXIL: [[VAR:%.*]] = call { i32, i32, i32, i32 } @llvm.dx.unpack.u8u32(i32 %{{.*}})
// CHECK-DXIL: [[extract1:%.*]] = extractvalue { i32, i32, i32, i32 } [[VAR]], 0
// CHECK-DXIL: [[insert1:%.*]] = insertelement <4 x i32> poison, i32 [[extract1]], i32 0
// CHECK-DXIL: [[extract2:%.*]] = extractvalue { i32, i32, i32, i32 } [[VAR]], 1
// CHECK-DXIL: [[insert2:%.*]] = insertelement <4 x i32> [[insert1]], i32 [[extract2]], i32 1
// CHECK-DXIL: [[extract3:%.*]] = extractvalue { i32, i32, i32, i32 } [[VAR]], 2
// CHECK-DXIL: [[insert3:%.*]] = insertelement <4 x i32> [[insert2]], i32 [[extract3]], i32 2
// CHECK-DXIL: [[extract4:%.*]] = extractvalue { i32, i32, i32, i32 } [[VAR]], 3
// CHECK-DXIL: [[insert4:%.*]] = insertelement <4 x i32> [[insert3]], i32 [[extract4]], i32 3
// CHECK-DXIL: ret <4 x i32> [[insert4]]
// CHECK-SPV: [[BITCAST:%.*]] = bitcast i32 {{%.*}} to <4 x i8>
// CHECK-SPV: [[EXTEND:%.*]] = zext <4 x i8> [[BITCAST]] to <4 x i32>
// CHECK-SPV: ret <4 x i32> [[EXTEND]]
uint32_t4 test_u8u32(uint8_t4_packed val) { return unpack_u8u32(val); }
