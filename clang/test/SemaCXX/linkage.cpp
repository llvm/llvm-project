// This is an IR generation test because the calculation of visibility
// during IR gen will cause linkage to be implicitly recomputed and
// compared against the earlier cached value.  If we had a way of
// testing linkage directly in Sema, that would be better.

// RUN: %clang_cc1 -Werror -Wno-non-c-typedef-for-linkage -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s

// CHECK: @_ZZN5test61A3fooEvE3bar = linkonce_odr global i32 0, align 4

// PR8926
namespace test0 {
  typedef struct {
    void *foo() { return 0; }
  } A;

  // CHECK: define linkonce_odr noundef i8* @_ZN5test01A3fooEv(

  void test(A *a) {
    a->foo();
  }
}

namespace test1 {
  typedef struct {
    template <unsigned n> void *foo() { return 0; }

    void foo() {
      foo<0>();
    }
  } A;

  // CHECK: define linkonce_odr void @_ZN5test11A3fooEv(
  // another at the end

  void test(A *a) {
    a->foo();
  }
}

namespace test2 {
  typedef struct {
    template <unsigned n> struct B {
      void *foo() { return 0; }
    };

    void foo(B<0> *b) {
      b->foo();
    }
  } A;

  // CHECK: define linkonce_odr void @_ZN5test21A3fooEPNS0_1BILj0EEE(

  void test(A *a) {
    a->foo(0);
  }
}

namespace test3 {
  namespace { struct A {}; }

  // CHECK: define internal void @_ZN5test34testENS_12_GLOBAL__N_11AE(
  void test(A a) {}
  void force() { test(A()); }

  // CHECK: define{{.*}} void @test3(
  extern "C" void test3(A a) {}
}

namespace {
  // CHECK: define{{.*}} void @test4(
  extern "C" void test4(void) {}
}

// PR9316: Ensure that even non-namespace-scope function declarations in
// a C declaration context respect that over the anonymous namespace.
extern "C" {
  namespace {
    struct X {
      int f() {
        extern int g();
        extern int a;

        // Test both for mangling in the code generation and warnings from use
        // of internal, undefined names via -Werror.
        // CHECK: call i32 @g(
        // CHECK: load i32, i32* @a,
        return g() + a;
      }
    };
  }
  // Force the above function to be emitted by codegen.
  int test(X& x) {
    return x.f();
  }
}

// CHECK: define linkonce_odr noundef i8* @_ZN5test11A3fooILj0EEEPvv(
// CHECK: define linkonce_odr noundef i8* @_ZN5test21A1BILj0EE3fooEv(

namespace test5 {
  struct foo {
  };
  extern "C" {
    const foo bar[]  = {
    };
  }
}

// Test that we don't compute linkage too hastily before we're done
// processing a record decl.  rdar://15928125
namespace test6 {
  typedef struct {
    int foo() {
      // Tested at top of file.
      static int bar = 0;
      return bar++;
    }
  } A;

  void test() {
    A a;
    a.foo();
  }
}
