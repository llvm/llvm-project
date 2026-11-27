// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -O2 -target-cpu x86-64-v3 -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -O2 -target-cpu x86-64-v3 -fclang-abi-compat=20 -o - | FileCheck --check-prefix CLANG-20 %s
// RUN: %clang_cc1 %s -triple x86_64-sie-ps4 -emit-llvm -O2 -target-cpu x86-64-v3 -o - | FileCheck --check-prefix CLANG-20 %s

using UInt64x2 = unsigned long long __attribute__((__vector_size__(16), may_alias));
using UInt64x4 = unsigned long long __attribute__((__vector_size__(32), may_alias));

template<int id>
struct XMM1 {
    UInt64x2 x;
};

struct XMM2 : XMM1<0>, XMM1<1> {
};

// CHECK: define{{.*}} @_Z3foov({{.*}} [[ARG:%.*]]){{.*}}
// CLANG-20: define{{.*}} <4 x double> @_Z3foov()
// CHECK: entry:
// CHECK-NEXT: store {{.*}}, ptr [[ARG]]{{.*}}
// CHECK-NEXT: [[TMP1:%.*]] = getelementptr {{.*}}, ptr [[ARG]]{{.*}}
// CHECK-NEXT: store {{.*}}, ptr [[TMP1]]{{.*}}
XMM2 foo() {
  XMM2 result;
  ((XMM1<0>*)&result)->x = UInt64x2{1, 2};
  ((XMM1<1>*)&result)->x = UInt64x2{3, 4};
  return result;
}

template<int id>
struct YMM1 {
    UInt64x4 x;
};

struct YMM2 : YMM1<0>, YMM1<1> {
};

// CHECK: define{{.*}} @_Z3barv({{.*}} [[ARG:%.*]]){{.*}}
// CLANG-20: define{{.*}} <8 x double> @_Z3barv()
// CHECK: entry:
// CHECK-NEXT: store {{.*}}, ptr [[ARG]]{{.*}}
// CHECK-NEXT: [[TMP1:%.*]] = getelementptr {{.*}}, ptr [[ARG]]{{.*}}
// CHECK-NEXT: store {{.*}}, ptr [[TMP1]]{{.*}}
YMM2 bar() {
  YMM2 result;
  ((YMM1<0>*)&result)->x = UInt64x4{1, 2, 3, 4};
  ((YMM1<1>*)&result)->x = UInt64x4{5, 6, 7, 8};
  return result;
}

// Test that empty base classes do not prevent structs with a single wide
// vector member from being passed/returned in registers (issue #203760).
struct EmptyBase {};

struct YMMWithEmptyBase : EmptyBase {
    UInt64x4 x;
};

// A struct with a single 256-bit vector and an empty base should use registers,
// matching the behavior with no base class.
// CHECK: define{{.*}} <4 x i64> @_Z18ymm_empty_base_retv()
// CLANG-20: define{{.*}} <4 x i64> @_Z18ymm_empty_base_retv()
YMMWithEmptyBase ymm_empty_base_ret() { return {}; }

// CHECK: define{{.*}} i64 @_Z19ymm_empty_base_pass16YMMWithEmptyBase(<4 x i64>
// CLANG-20: define{{.*}} i64 @_Z19ymm_empty_base_pass16YMMWithEmptyBase(<4 x i64>
unsigned long long ymm_empty_base_pass(YMMWithEmptyBase x) { return x.x[0]; }
