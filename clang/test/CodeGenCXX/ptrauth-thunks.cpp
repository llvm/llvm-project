// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -emit-llvm -std=c++11 %s -o - -O1 | FileCheck %s

namespace Test1 {
  struct B1 {
    virtual void* foo1() {
      return 0;
    }
  };
  struct Pad1 {
    virtual ~Pad1() {}
  };
  struct Proxy1 : Pad1, B1 {
    virtual ~Proxy1() {}
  };
  struct D : virtual Proxy1 {
    virtual ~D() {}
    virtual void* foo1();
  };
  void* D::foo1() {
    return (void*)this;
  }
}

// CHECK-LABEL: define linkonce_odr void @_ZTv0_n24_N5Test11DD0Ev(%"struct.Test1::D"* %this)
// CHECK: %[[BitcastThis:.*]] = bitcast %"struct.Test1::D"* %this to i64*
// CHECK: %[[SignedVTable:.*]] = load i64, i64* %[[BitcastThis]], align 8
// CHECK: %[[VTable:.*]] = tail call i64 @llvm.ptrauth.auth.i64(i64 %[[SignedVTable]], i32 2, i64 0)