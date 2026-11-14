// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-int16-type -fnative-half-type -o - | \
// RUN:  FileCheck %s -check-prefix=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-int16-type -fnative-half-type -o - | \
// RUN:  FileCheck %s -check-prefix=CHECK,CHECK-SPV

// CHECK: define {{.*}} <4 x i16> @_Z10test_s8s16
// CHECK-DXIL: [[VAR:%.*]] = call { i16, i16, i16, i16 } @llvm.dx.unpack.s8s16(i32 %{{.*}})
// CHECK-DXIL: [[extract1:%.*]] = extractvalue { i16, i16, i16, i16 } [[VAR]], 0
// CHECK-DXIL: [[insert1:%.*]] = insertelement <4 x i16> poison, i16 [[extract1]], i32 0
// CHECK-DXIL: [[extract2:%.*]] = extractvalue { i16, i16, i16, i16 } [[VAR]], 1
// CHECK-DXIL: [[insert2:%.*]] = insertelement <4 x i16> [[insert1]], i16 [[extract2]], i32 1
// CHECK-DXIL: [[extract3:%.*]] = extractvalue { i16, i16, i16, i16 } [[VAR]], 2
// CHECK-DXIL: [[insert3:%.*]] = insertelement <4 x i16> [[insert2]], i16 [[extract3]], i32 2
// CHECK-DXIL: [[extract4:%.*]] = extractvalue { i16, i16, i16, i16 } [[VAR]], 3
// CHECK-DXIL: [[insert4:%.*]] = insertelement <4 x i16> [[insert3]], i16 [[extract4]], i32 3
// CHECK-DXIL: ret <4 x i16> [[insert4]]
// CHECK-SPV: [[BITCAST:%.*]] = bitcast i32 {{%.*}} to <4 x i8>
// CHECK-SPV: [[EXTEND:%.*]] = sext <4 x i8> [[BITCAST]] to <4 x i16>
// CHECK-SPV: ret <4 x i16> [[EXTEND]]
int16_t4 test_s8s16(int8_t4_packed val) { return unpack_s8s16(val); }
