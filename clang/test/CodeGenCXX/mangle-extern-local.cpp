// RUN: %clang_cc1 %s -triple i386-unknown-unknown -emit-llvm -o - | FileCheck %s

// CHECK: @var1 = external global i32
// CHECK: @_ZN1N4var2E = external global i32
// CHECK: @var5 = external global i32
// CHECK: @_ZN1N4var3E = external global i32
// CHECK: @_ZN1N4var4E = external global i32

// CHECK: declare i32 @_Z5func1v()
// CHECK: declare i32 @_ZN1N5func2Ev()
// CHECK: declare i32 @func4()
// CHECK: declare i32 @_ZN1N5func3Ev()

int f1() {
  extern int var1, func1();
  return var1 + func1();
}

namespace N {

int f2() {
  extern int var2, func2();
  return var2 + func2();
}

struct S {
  static int f3() {
    extern int var3, func3();
    struct LC { int localfunc() { extern int var4; return var4; } };
    LC localobj;
    return var3 + func3() + localobj.localfunc();
  }
};

int anchorf3() { return S::f3(); } 

extern "C" {
int f4() {
  extern int var5, func4();
  return var5 + func4();
}
}

}

