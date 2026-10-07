// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-int16-type -fnative-half-type -o - | \
// RUN:  FileCheck %s -check-prefix=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-int16-type -fnative-half-type -o - | \
// RUN:  FileCheck %s -check-prefix=CHECK,CHECK-SPV

// CHECK: define {{.*}} <4 x i16> @_Z10test_u8u16u15uint8_t4_packed
// CHECK-DXIL: [[VAR:%.*]] = call { i16, i16, i16, i16 } @llvm.dx.unpack.u8u16(i32 %{{.*}})
// CHECK-DXIL: %{{.*}} = extractvalue { i16, i16, i16, i16 } [[VAR]], 0
// CHECK-DXIL: %{{.*}} = insertelement <4 x i16>
// CHECK-DXIL: %{{.*}} = extractvalue { i16, i16, i16, i16 } [[VAR]], 1
// CHECK-DXIL: %{{.*}} = insertelement <4 x i16>
// CHECK-DXIL: %{{.*}} = extractvalue { i16, i16, i16, i16 } [[VAR]], 2
// CHECK-DXIL: %{{.*}} = insertelement <4 x i16>
// CHECK-DXIL: %{{.*}} = extractvalue { i16, i16, i16, i16 } [[VAR]], 3
// CHECK-DXIL: %{{.*}} = insertelement <4 x i16>
// CHECK-DXIL: ret <4 x i16>
// CHECK-SPV: [[VAR:%.*]] = call <4 x i16> @llvm.spv.unpack.u8u16(i32 %{{.*}})
// CHECK-SPV: ret <4 x i16> [[VAR]]
uint16_t4 test_u8u16(uint8_t4_packed val) { return unpack_u8u16(val); }
