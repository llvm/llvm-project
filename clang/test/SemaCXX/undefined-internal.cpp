// RUN: %clang_cc1 -fsyntax-only -verify %s

// Make sure we don't produce invalid IR.
// RUN: %clang_cc1 -emit-llvm-only %s

namespace test1 {
  static void foo(); // expected-warning {{function 'test1::foo' has internal linkage but is not defined}}
  template <class T> static void bar(); // expected-warning {{function 'test1::bar<int>' has internal linkage but is not defined}}

  void test() {
    foo(); // expected-note {{used here}}
    bar<int>(); // expected-note {{used here}}
  }
}

namespace test2 {
  namespace {
    void foo(); // expected-warning {{function 'test2::<anonymous namespace>::foo' has internal linkage but is not defined}}
    extern int var; // expected-warning {{variable 'test2::<anonymous namespace>::var' has internal linkage but is not defined}}
    template <class T> void bar(); // expected-warning {{function 'test2::<anonymous namespace>::bar<int>' has internal linkage but is not defined}}
  }
  void test() {
    foo(); // expected-note {{used here}}
    var = 0; // expected-note {{used here}}
    bar<int>(); // expected-note {{used here}}
  }
}

namespace test3 {
  namespace {
    void foo();
    extern int var;
    template <class T> void bar();
  }

  void test() {
    foo();
    var = 0;
    bar<int>();
  }

  namespace {
    void foo() {}
    int var = 0;
    template <class T> void bar() {}
  }
}

namespace test4 {
  namespace {
    struct A {
      A(); // expected-warning {{function 'test4::<anonymous namespace>::A::A' has internal linkage but is not defined}}
      ~A();// expected-warning {{function 'test4::<anonymous namespace>::A::~A' has internal linkage but is not defined}}
      virtual void foo(); // expected-warning {{function 'test4::<anonymous namespace>::A::foo' has internal linkage but is not defined}}
      virtual void bar() = 0;
      virtual void baz(); // expected-warning {{function 'test4::<anonymous namespace>::A::baz' has internal linkage but is not defined}}
    };
  }

  void test(A &a) {
    a.foo(); // expected-note {{used here}}
    a.bar();
    a.baz(); // expected-note {{used here}}
  }

  struct Test : A {
    Test() {} // expected-note 2 {{used here}}
  };
}

// rdar://problem/9014651
namespace test5 {
  namespace {
    struct A {};
  }

  template <class N> struct B {
    static int var; // expected-warning {{variable 'test5::B<test5::<anonymous>::A>::var' has internal linkage but is not defined}}
    static void foo(); // expected-warning {{function 'test5::B<test5::<anonymous>::A>::foo' has internal linkage but is not defined}}
  };

  void test() {
    B<A>::var = 0; // expected-note {{used here}}
    B<A>::foo(); // expected-note {{used here}}
  }
}

namespace test6 {
  template <class T> struct A {
    static const int zero = 0;
    static const int one = 1;
    static const int two = 2;

    int value;

    A() : value(zero) {
      value = one;
    }
  };

  namespace { struct Internal; }

  void test() {
    A<Internal> a;
    a.value = A<Internal>::two;
  }
}

// We support (as an extension) private, undefined copy constructors when
// a temporary is bound to a reference even in C++98. Similarly, we shouldn't
// warn about this copy constructor being used without a definition.
namespace PR9323 {
  namespace {
    struct Uncopyable {
      Uncopyable() {}
    private:
      Uncopyable(const Uncopyable&); // expected-note {{declared private here}}
    };
  }
  void f(const Uncopyable&) {}
  void test() {
    f(Uncopyable()); // expected-warning {{C++98 requires an accessible copy constructor}}
  };
}


namespace std { class type_info; };
namespace cxx11_odr_rules {
  // Note: the way this test is written isn't really ideal, but there really
  // isn't any other way to check that the odr-used logic for constants
  // is working without working implicit capture in lambda-expressions.
  // (The more accurate used-but-not-defined warning is the only other visible
  // effect of accurate odr-used computation.)
  //
  // Note that the warning in question can trigger in cases some people would
  // consider false positives; hopefully that happens rarely in practice.
  //
  // FIXME: Suppressing this test while I figure out how to fix a bug in the
  // odr-use marking code.

  namespace {
    struct A {
      static const int unused = 10;
      static const int used1 = 20; // xpected-warning {{internal linkage}}
      static const int used2 = 20; // xpected-warning {{internal linkage}}
      virtual ~A() {}
    };
  }

  void a(int,int);
  A& p(const int&) { static A a; return a; }

  // Check handling of default arguments
  void b(int = A::unused);

  void tests() {
    // Basic test
    a(A::unused, A::unused);

    // Check that nesting an unevaluated or constant-evaluated context does
    // the right thing.
    a(A::unused, sizeof(int[10]));

    // Check that the checks work with unevaluated contexts
    (void)sizeof(p(A::used1));
    (void)typeid(p(A::used1)); // xpected-note {{used here}}

    // Misc other testing
    a(A::unused, 1 ? A::used2 : A::used2); // xpected-note {{used here}}
    b();
  }
}


namespace OverloadUse {
  namespace {
    void f();
    void f(int); // expected-warning {{function 'OverloadUse::<anonymous namespace>::f' has internal linkage but is not defined}}
  }
  template<void x()> void t(int*) { x(); }
  template<void x(int)> void t(long*) { x(10); } // expected-note {{used here}}
  void g() { long a; t<f>(&a); }
}
