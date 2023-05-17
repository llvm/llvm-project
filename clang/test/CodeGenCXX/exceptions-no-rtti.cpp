// RUN: %clang_cc1 -fno-rtti -fcxx-exceptions -fexceptions -Wno-dynamic-exception-spec %s -triple=x86_64-apple-darwin10 -emit-llvm -o - %std_cxx98- | FileCheck %s

// CHECK: @_ZTIN5test11AE = linkonce_odr constant
// CHECK: @_ZTIN5test11BE = linkonce_odr constant
// CHECK: @_ZTIN5test11CE = linkonce_odr constant
// CHECK: @_ZTIN5test11DE = linkonce_odr constant
// CHECK: @_ZTIPN5test11DE = linkonce_odr constant {{.*}} @_ZTIN5test11DE

// PR6974: this shouldn't crash
namespace test0 {
  class err {};

  void f(void) {
    try {
    } catch (err &) {
    }
  }
}

namespace test1 {
  // These classes have key functions defined out-of-line.  Under
  // normal circumstances, we wouldn't generate RTTI for them; under
  // -fno-rtti, we generate RTTI only when required by EH.  But
  // everything gets hidden visibility because we assume that all
  // users are also compiled under -fno-rtti and therefore will be
  // emitting RTTI regardless of key function.
  class A { virtual void foo(); };
  class B { virtual void foo(); };
  class C { virtual void foo(); };
  class D { virtual void foo(); };

  void opaque();

  void test0() {
    throw A();
  }

  void test1() throw(B) {
    opaque();
  }

  void test2() {
    try {
      opaque();
    } catch (C&) {}
  }

  void test3(D *ptr) {
    throw ptr;
  };
}
