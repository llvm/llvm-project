// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-int16-type -fnative-half-type -o - | \
// RUN:  FileCheck %s -DCALL=dx
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-int16-type -fnative-half-type -o - | \
// RUN:  FileCheck %s -DCALL=spv

// CHECK-LABEL: define {{.*}} i32 @_Z8test_u16Dv4_t
// CHECK: [[VAR:%.*]] = call i32 @llvm.[[CALL]].pack.u8.v4i16(<4 x i16> %{{.*}})
// CHECK-NEXT: ret i32 [[VAR]]
uint8_t4_packed test_u16(uint16_t4 val) { return pack_u8(val); }

// CHECK-LABEL: define {{.*}} i32 @_Z8test_u32Dv4_j
// CHECK: [[VAR:%.*]] = call i32 @llvm.[[CALL]].pack.u8.v4i32(<4 x i32> %{{.*}})
// CHECK-NEXT: ret i32 [[VAR]]
uint8_t4_packed test_u32(uint4 val) { return pack_u8(val); }
