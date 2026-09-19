// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-int16-type -fnative-half-type -o - | \
// RUN:  FileCheck %s -DCALL=dx
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple spirv-pc-vulkan-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -fnative-int16-type -fnative-half-type -o - | \
// RUN:  FileCheck %s -DCALL=spv

// CHECK-LABEL: define {{.*}} i32 @_Z8test_s16Dv4_s
// CHECK: [[VAR:%.*]] = call i32 @llvm.[[CALL]].pack.s8.v4i16(<4 x i16> %{{.*}})
// CHECK-NEXT: ret i32 [[VAR]]
int8_t4_packed test_s16(int16_t4 val) { return pack_s8(val); }

// CHECK-LABEL: define {{.*}} i32 @_Z8test_s32Dv4_i
// CHECK: [[VAR:%.*]] = call i32 @llvm.[[CALL]].pack.s8.v4i32(<4 x i32> %{{.*}})
// CHECK-NEXT: ret i32 [[VAR]]
int8_t4_packed test_s32(int4 val) { return pack_s8(val); }
