// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// Regression test for issue llvm/llvm-project#213814

// CHECK-LABEL: test_lt
// NATIVE_HALF: [[CMP:%.*]] = fcmp {{.*}} olt <4 x half>
// NATIVE_HALF-NEXT: [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i16>
// NATIVE_HALF-NEXT: [[RET:%.*]] = sext <4 x i16> [[SEXT]] to <4 x i32>
// NO_HALF: [[CMP:%.*]] = fcmp {{.*}} olt <4 x float>
// NO_HALF-NEXT: [[RET:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
// CHECK-NEXT: ret <4 x i32> [[RET]]
int4 test_lt(half4 a, half4 b) {
    return a < b;
}

// CHECK-LABEL: test_le
// NATIVE_HALF: [[CMP:%.*]] = fcmp {{.*}} ole <4 x half>
// NATIVE_HALF-NEXT: [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i16>
// NATIVE_HALF-NEXT: [[RET:%.*]] = sext <4 x i16> [[SEXT]] to <4 x i32>
// NO_HALF: [[CMP:%.*]] = fcmp {{.*}} ole <4 x float>
// NO_HALF-NEXT: [[RET:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
// CHECK-NEXT: ret <4 x i32> [[RET]]
int4 test_le(half4 a, half4 b) {
    return a <= b;
}

// CHECK-LABEL: test_gt
// NATIVE_HALF: [[CMP:%.*]] = fcmp {{.*}} ogt <4 x half>
// NATIVE_HALF-NEXT: [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i16>
// NATIVE_HALF-NEXT: [[RET:%.*]] = sext <4 x i16> [[SEXT]] to <4 x i32>
// NO_HALF: [[CMP:%.*]] = fcmp {{.*}} ogt <4 x float>
// NO_HALF-NEXT: [[RET:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
// CHECK-NEXT: ret <4 x i32> [[RET]]
int4 test_gt(half4 a, half4 b) {
    return a > b;
}

// CHECK-LABEL: test_ge
// NATIVE_HALF: [[CMP:%.*]] = fcmp {{.*}} oge <4 x half>
// NATIVE_HALF-NEXT: [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i16>
// NATIVE_HALF-NEXT: [[RET:%.*]] = sext <4 x i16> [[SEXT]] to <4 x i32>
// NO_HALF: [[CMP:%.*]] = fcmp {{.*}} oge <4 x float>
// NO_HALF-NEXT: [[RET:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
// CHECK-NEXT: ret <4 x i32> [[RET]]
int4 test_ge(half4 a, half4 b) {
    return a >= b;
}

// CHECK-LABEL: test_eq
// NATIVE_HALF: [[CMP:%.*]] = fcmp {{.*}} oeq <4 x half>
// NATIVE_HALF-NEXT: [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i16>
// NATIVE_HALF-NEXT: [[RET:%.*]] = sext <4 x i16> [[SEXT]] to <4 x i32>
// NO_HALF: [[CMP:%.*]] = fcmp {{.*}} oeq <4 x float>
// NO_HALF-NEXT: [[RET:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
// CHECK-NEXT: ret <4 x i32> [[RET]]
int4 test_eq(half4 a, half4 b) {
    return a == b;
}

// CHECK-LABEL: test_ne
// NATIVE_HALF: [[CMP:%.*]] = fcmp {{.*}} une <4 x half>
// NATIVE_HALF-NEXT: [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i16>
// NATIVE_HALF-NEXT: [[RET:%.*]] = sext <4 x i16> [[SEXT]] to <4 x i32>
// NO_HALF: [[CMP:%.*]] = fcmp {{.*}} une <4 x float>
// NO_HALF-NEXT: [[RET:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
// CHECK-NEXT: ret <4 x i32> [[RET]]
int4 test_ne(half4 a, half4 b) {
    return a != b;
}
