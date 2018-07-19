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

// CHECK: @_ZTV1S = linkonce_odr unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI1S to i8*), i8* bitcast (i32 (%struct.S*)* @_ZN1S4testEv to i8*)] }

// CHECK-LABEL: define i32 @_Z2fnv()
// CHECK:   call x86_thiscallcc void @_ZN1SC1Ev(%struct.S* %s)
// CHECK:   %{{[.0-9A-Z_a-z]+}} = call x86_thiscallcc i32 @_ZN1S4testEv(%struct.S* %s)

// CHECK-LABEL: define linkonce_odr x86_thiscallcc void @_ZN1SC1Ev(%struct.S* %this)
// CHECK:   call x86_thiscallcc void @_ZN1SC2Ev(%struct.S* %{{[.0-9A-Z_a-z]+}})

// CHECK-LABEL: define linkonce_odr x86_thiscallcc i32 @_ZN1S4testEv(%struct.S* %this)
// CHECK:   %{{[.0-9A-Z_a-z]+}} = call x86_thiscallcc i32 @_ZN1I4testEv(%__interface.I* %{{[.0-9A-Z_a-z]+}})

// CHECK-LABEL: define linkonce_odr x86_thiscallcc void @_ZN1SC2Ev(%struct.S* %this)
// CHECK:   call x86_thiscallcc void @_ZN1IC2Ev(%__interface.I* %{{[.0-9A-Z_a-z]+}})
// CHECK:   store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV1S, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** %{{[.0-9A-Z_a-z]+}}

// CHECK-LABEL: define linkonce_odr x86_thiscallcc void @_ZN1IC2Ev(%__interface.I* %this)
// CHECK:   store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV1I, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** %{{[.0-9A-Z_a-z]+}}

// CHECK-LABEL: define linkonce_odr x86_thiscallcc i32 @_ZN1I4testEv(%__interface.I* %this)
// CHECK:   ret i32 1
