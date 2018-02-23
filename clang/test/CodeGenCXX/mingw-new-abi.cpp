// RUN: %clang_cc1 -emit-llvm -triple i386-pc-mingw32 %s -o - | FileCheck --check-prefix=MINGW %s
// RUN: %clang_cc1 -emit-llvm -triple i386-pc-cygwin %s -o - | FileCheck --check-prefix=CYGWIN %s

namespace test1 {
  struct foo {
    //  MINGW: declare dso_local x86_thiscallcc void @_ZN5test13foo1fEv
    //  CYGWIN: declare dso_local void @_ZN5test13foo1fEv
    void f();
  };
  void g(foo *x) {
    x->f();
  }
}
