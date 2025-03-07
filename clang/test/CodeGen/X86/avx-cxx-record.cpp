// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -O2 -target-cpu x86-64-v3 -o - | FileCheck %s

using UInt64x2 = unsigned long long __attribute__((__vector_size__(16), may_alias));

template<int id>
struct XMM1 {
    UInt64x2 x;
};

struct XMM2 : XMM1<0>, XMM1<1> {
};

// CHECK: define{{.*}} @_Z3foov({{.*}} [[ARG:%.*]]){{.*}}
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
