// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s

#pragma GCC visibility push(hidden)
struct x {
  static int y;
};
#pragma GCC visibility pop
int x::y = 10;
// CHECK: @_ZN1x1yE = hidden global

#pragma GCC visibility push(hidden)
struct __attribute((visibility("default"))) x2 {
  static int y;
};
int x2::y = 10;
// CHECK: @_ZN2x21yE ={{.*}} global
#pragma GCC visibility pop

#pragma GCC visibility push(hidden)
template<class T> struct x4 {
  static int y;
};
#pragma GCC visibility pop
template<> int x4<int>::y = 10;
// CHECK: @_ZN2x4IiE1yE = hidden global i32

#pragma GCC visibility push(hidden)
template<int x> int f() { return x; }
extern "C" int g() { return f<3>(); }
#pragma GCC visibility pop
// CHECK-LABEL: define hidden i32 @g()
// CHECK-LABEL: define linkonce_odr hidden i32 @_Z1fILi3EEiv()

#pragma GCC visibility push(hidden)
template<class T> struct x5 {
  void y();
};
#pragma GCC visibility pop
template<> void x5<int>::y() {}
// CHECK-LABEL: define hidden void @_ZN2x5IiE1yEv

#pragma GCC visibility push(hidden)
namespace n __attribute((visibility("default"))) {
  void f() {}
  // CHECK-LABEL: define{{.*}} void @_ZN1n1fEv
}
#pragma GCC visibility pop

namespace n __attribute((visibility("default")))  {
#pragma GCC visibility push(hidden)
  void g() {}
  // CHECK-LABEL: define hidden void @_ZN1n1gEv
#pragma GCC visibility pop
}

namespace test2 {
#pragma GCC visibility push(default)
#pragma GCC visibility push(hidden)
  struct foo { // foo is hidden
  };
#pragma GCC visibility pop
  struct foo; // declaration is ok, we ignore the default in the stack
  template<typename T>
  struct bar { // bar is default
    static void f(){}
  };
#pragma GCC visibility pop
  void zed() {
    bar<foo>::f();
    bar<int>::f();
  }
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN5test23barINS_3fooEE1fEv
  // CHECK-LABEL: define linkonce_odr void @_ZN5test23barIiE1fEv
}
