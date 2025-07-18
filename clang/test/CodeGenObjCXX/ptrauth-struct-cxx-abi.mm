// RUN: %clang_cc1 -triple arm64-apple-ios11 -fptrauth-calls -fptrauth-intrinsics -std=c++11 -fobjc-arc -emit-llvm -o - %s | FileCheck %s

#define AQ __ptrauth(1,1,50)

struct AddrDiscStrong0 {
  int * AQ f0; // Signed using address discrimination.
  __strong id f1;
};

struct AddrDiscStrong1 {
  AddrDiscStrong1(const AddrDiscStrong1 &);
  int * AQ f0; // Signed using address discrimination.
  __strong id f1;
};

// Check that AddrDiscStrong0 is destructed in the callee.

// CHECK: define void @_Z24testParamAddrDiscStrong015AddrDiscStrong0(ptr dead_on_return noundef %[[A:.*]])
// CHECK: call noundef ptr @_ZN15AddrDiscStrong0D1Ev(ptr noundef nonnull align {{[0-9]+}} dereferenceable(16) %[[A]])
// CHECK: ret void

// CHECK: define linkonce_odr noundef ptr @_ZN15AddrDiscStrong0D1Ev(

void testParamAddrDiscStrong0(AddrDiscStrong0 a) {
}

// Check that AddrDiscStrong1 is not destructed in the callee because it has a
// non-trivial copy constructor.

// CHECK: define void @_Z24testParamAddrDiscStrong115AddrDiscStrong1(ptr noundef %{{.*}})
// CHECK-NOT: call
// CHECK: ret void

void testParamAddrDiscStrong1(AddrDiscStrong1 a) {
}
