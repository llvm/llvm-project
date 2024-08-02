// RUN: %clang_cc1 -std=c++11 -fms-extensions -Wno-microsoft -triple=i386-pc-windows-gnu -emit-llvm %s -o - | FileCheck %s

__interface I {
  int test() {
    return 1;
  }
};

struct S : I {
  virtual int test() override {
    return I::test();
  }
};

int fn() {
  S s;
  return s.test();
}

// CHECK: @_ZTV1S = linkonce_odr dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1S, ptr @_ZN1S4testEv] }

// CHECK-LABEL: define dso_local noundef i32 @_Z2fnv()
// CHECK:   call x86_thiscallcc void @_ZN1SC1Ev(ptr {{[^,]*}} %s)
// CHECK:   %{{[.0-9A-Z_a-z]+}} = call x86_thiscallcc noundef i32 @_ZN1S4testEv(ptr {{[^,]*}} %s)

// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc void @_ZN1SC1Ev(ptr {{[^,]*}} %this)
// CHECK:   call x86_thiscallcc void @_ZN1SC2Ev(ptr {{[^,]*}} %{{[.0-9A-Z_a-z]+}})

// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc noundef i32 @_ZN1S4testEv(ptr {{[^,]*}} %this)
// CHECK:   %{{[.0-9A-Z_a-z]+}} = call x86_thiscallcc noundef i32 @_ZN1I4testEv(ptr {{[^,]*}} %{{[.0-9A-Z_a-z]+}})

// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc void @_ZN1SC2Ev(ptr {{[^,]*}} %this)
// CHECK:   call x86_thiscallcc void @_ZN1IC2Ev(ptr {{[^,]*}} %{{[.0-9A-Z_a-z]+}})
// CHECK:   store ptr getelementptr inbounds inrange(-8, 4) ({ [3 x ptr] }, ptr @_ZTV1S, i32 0, i32 0, i32 2), ptr %{{[.0-9A-Z_a-z]+}}

// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc void @_ZN1IC2Ev(ptr {{[^,]*}} %this)
// CHECK:   store ptr getelementptr inbounds inrange(-8, 4) ({ [3 x ptr] }, ptr @_ZTV1I, i32 0, i32 0, i32 2), ptr %{{[.0-9A-Z_a-z]+}}

// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc noundef i32 @_ZN1I4testEv(ptr {{[^,]*}} %this)
// CHECK:   ret i32 1
