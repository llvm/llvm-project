// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// An extra byte should be allocated for an empty class.
namespace Test1 {
  // CHECK: %"struct.Test1::A" = type { i8 }
  struct A { } a;
}

namespace Test2 {
  // No need to add tail padding here.
  // CHECK: %"struct.Test2::A" = type { ptr, i32 }
  struct A { void *a; int b; } a;
}

namespace Test3 {
  // C should have a vtable pointer.
  // CHECK: %"struct.Test3::A" = type <{ ptr, i32, [4 x i8] }>
  struct A { virtual void f(); int a; } a;
}

namespace Test4 {
  // Test from PR5589.
  // CHECK: %"struct.Test4::B" = type { %"struct.Test4::A", i16, double }
  // CHECK: %"struct.Test4::A" = type { i32, i8, float }
  struct A {
    int a;
    char c;
    float b;
  };
  struct B : public A {
    short d;
    double e;
  } b;
}

namespace Test5 {
  struct A {
    virtual void f();
    char a;
  };

  // CHECK: %"struct.Test5::B" = type {  %"struct.Test5::A.base", i8, i8, [5 x i8] }
  struct B : A {
    char b : 1;
    char c;
  } b;
}

// PR10912: don't crash
namespace Test6 {
  template <typename T> class A {
    // If T is complete, IR-gen will want to translate it recursively
    // when translating T*.
    T *foo;
  };

  class B;

  // This causes IR-gen to have an incomplete translation of A<B>
  // sitting around.
  A<B> *a;

  class C {};
  class B : public C {
    // This forces Sema to instantiate A<B>, which triggers a callback
    // to IR-gen.  Because of the previous, incomplete translation,
    // IR-gen actually cares, and it immediately tries to complete
    // A<B>'s IR type.  That, in turn, causes the translation of B*.
    // B isn't complete yet, but it has a definition, and if we try to
    // compute a record layout for that definition then we'll really
    // regret it later.
    A<B> a;
  };

  // The derived class E and empty base class C are required to
  // provoke the original assertion.
  class E : public B {};
  E *e;
}

// Make sure this doesn't crash. (It's okay if we start rejecting it at some
// point.)
namespace Test7 {
  #pragma pack (1)
  class A {};
  // CHECK: %"class.Test7::B" = type <{ ptr, i8 }>
  class B {
     virtual ~B();
     A a;
  };
  B test(B b) { return b; }
  #pragma pack ()
}

// Shouldn't crash.
namespace Test8 {
  struct A {};
  struct D { int a; };
  struct B : virtual D, A { };
  struct C : B, A { void f() {} };
  C c;
}
