// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -disable-llvm-passes -emit-llvm -finclude-default-header -fnative-half-type -o - %s | FileCheck %s

// CHECK-LABEL: case1
// CHECK: [[ToBool:%.*]] = icmp ne <2 x i32> {{.*}}, zeroinitializer
// CHECK-NEXT: [[BoolCmp:%.*]] = icmp eq <2 x i1> [[ToBool]], zeroinitializer
// CHECK-NEXT: {{.*}} = zext <2 x i1> [[BoolCmp]] to <2 x i32>
export uint32_t2 case1(uint32_t2 b) {
    return !b;
}

// CHECK-LABEL: case2
// CHECK: [[ToBool:%.*]] = icmp ne <3 x i32> {{.*}}, zeroinitializer
// CHECK-NEXT: [[BoolCmp:%.*]] = icmp eq <3 x i1> [[ToBool]], zeroinitializer
// CHECK-NEXT: {{.*}} = zext <3 x i1> [[BoolCmp]] to <3 x i32>
export int32_t3 case2(int32_t3 b) {
    return !b;
}

// CHECK-LABEL: case3
// CHECK: [[ToBool:%.*]] = fcmp reassoc nnan ninf nsz arcp afn une half {{.*}}, 0xH0000
// CHECK-NEXT: [[BoolCmp:%.*]] = xor i1 [[ToBool]], true
// CHECK-NEXT: {{.*}} = uitofp i1 [[BoolCmp]] to half
export float16_t case3(float16_t b) {
    return !b;
}

// CHECK-LABEL: case4
// CHECK: [[ToBool:%.*]] = fcmp reassoc nnan ninf nsz arcp afn une <4 x float> {{.*}}, zeroinitializer
// CHECK-NEXT: [[BoolCmp:%.*]] = icmp eq <4 x i1> [[ToBool]], zeroinitializer
// CHECK-NEXT: {{.*}} = uitofp <4 x i1> [[BoolCmp]] to <4 x float>
export float4 case4(float4 b) {
    return !b;
}
