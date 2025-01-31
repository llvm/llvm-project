// RUN: %clang_cc1 -fsyntax-only -verify -Wbind-to-temporary-copy %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wbind-to-temporary-copy -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wbind-to-temporary-copy -std=c++11 %s

// RUN: %clang_cc1 -fsyntax-only -verify -Wbind-to-temporary-copy %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -fsyntax-only -verify -Wbind-to-temporary-copy -std=c++98 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -fsyntax-only -verify -Wbind-to-temporary-copy -std=c++11 %s -fexperimental-new-constant-interpreter


// Make sure we don't produce invalid IR.
// RUN: %clang_cc1 -emit-llvm-only %s
// RUN: %clang_cc1 -emit-llvm-only -std=c++98 %s
// RUN: %clang_cc1 -emit-llvm-only -std=c++11 %s

// RUN: %clang_cc1 -emit-llvm-only %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -emit-llvm-only -std=c++98 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -emit-llvm-only -std=c++11 %s -fexperimental-new-constant-interpreter


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
    void foo(); // expected-warning {{function 'test2::(anonymous namespace)::foo' has internal linkage but is not defined}}
    extern int var; // expected-warning {{variable 'test2::(anonymous namespace)::var' has internal linkage but is not defined}}
    template <class T> void bar(); // expected-warning {{function 'test2::(anonymous namespace)::bar<int>' has internal linkage but is not defined}}
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
      A(); // expected-warning {{function 'test4::(anonymous namespace)::A::A' has internal linkage but is not defined}}
      ~A();// expected-warning {{function 'test4::(anonymous namespace)::A::~A' has internal linkage but is not defined}}
      virtual void foo(); // expected-warning {{function 'test4::(anonymous namespace)::A::foo' has internal linkage but is not defined}}
      virtual void bar() = 0;
      virtual void baz(); // expected-warning {{function 'test4::(anonymous namespace)::A::baz' has internal linkage but is not defined}}
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

namespace test5 {
  namespace {
    struct A {};
  }

  template <class N> struct B {
    static int var; // expected-warning {{variable 'test5::B<test5::(anonymous namespace)::A>::var' has internal linkage but is not defined}}
    static void foo(); // expected-warning {{function 'test5::B<test5::(anonymous namespace)::A>::foo' has internal linkage but is not defined}}
  };
  extern template int B<A>::var;

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
    f(Uncopyable());
#if __cplusplus <= 199711L // C++03 or earlier modes
    // expected-warning@-2 {{C++98 requires an accessible copy constructor}}
#else
    // expected-warning@-4 {{copying parameter of type 'Uncopyable' when binding a reference to a temporary would invoke an inaccessible constructor in C++98}}
#endif
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
    (void)typeid(p(A::used1)); // expected-warning {{expression with side effects will be evaluated despite being used as an operand to 'typeid'}} xpected-note {{used here}}

    // Misc other testing
    a(A::unused, 1 ? A::used2 : A::used2); // xpected-note {{used here}}
    b();
  }
}


namespace OverloadUse {
  namespace {
    void f();
    void f(int); // expected-warning {{function 'OverloadUse::(anonymous namespace)::f' has internal linkage but is not defined}}
    void f(int, int); // expected-warning {{function 'OverloadUse::(anonymous namespace)::f' has internal linkage but is not defined}}
#if __cplusplus < 201103L
    // expected-note@-3 {{here}}
    // expected-note@-3 {{here}}
#endif
  }
  template<void x()> void t() { x(); }
  template<void x(int)> void t(int*) { x(10); }
  template<void x(int, int)> void t(int*, int*) {}
  void g(int n) {
    t<f>(&n); // expected-note {{used here}}
    t<f>(&n, &n); // expected-note {{used here}}
#if __cplusplus < 201103L
    // expected-warning@-3 {{non-type template argument referring to function 'f' with internal linkage}}
    // expected-note@-7 {{template parameter is declared here}}
    // expected-note@-5 {{while substituting explicitly-specified template arguments}}
    // expected-warning@-5 {{non-type template argument referring to function 'f' with internal linkage}}
    // expected-note@-9 {{template parameter is declared here}}
    // expected-note@-7 {{while substituting explicitly-specified template arguments}}
#endif
  }
}

namespace test7 {
  typedef struct { // expected-warning {{add a tag name}}
    void bar(); // expected-note {{this member}}
    void foo() {
      bar();
    }
  } A; // expected-note {{this typedef}}
}

namespace test8 {
  typedef struct {
    void bar(); // expected-warning {{function 'test8::(anonymous struct)::bar' has internal linkage but is not defined}}
    void foo() {
      bar(); // expected-note {{used here}}
    }
  } *A;
}

namespace test9 {
  namespace {
    struct X {
      virtual void notused() = 0;
      virtual void used() = 0; // expected-warning {{function 'test9::(anonymous namespace)::X::used' has internal linkage but is not defined}}
    };
  }
  void test(X &x) {
    x.notused();
    x.X::used(); // expected-note {{used here}}
  }
}

namespace test10 {
  namespace {
    struct X {
      virtual void notused() = 0;
      virtual void used() = 0; // expected-warning {{function 'test10::(anonymous namespace)::X::used' has internal linkage but is not defined}}

      void test() {
        notused();
        (void)&X::notused;
        (this->*&X::notused)();
        X::used();  // expected-note {{used here}}
      }
    };
    struct Y : X {
      using X::notused;
    };
  }
}

namespace test11 {
  namespace {
    struct A {
      virtual bool operator()() const = 0;
      virtual void operator!() const = 0;
      virtual bool operator+(const A&) const = 0;
      virtual int operator[](int) const = 0;
      virtual const A* operator->() const = 0;
      int member;
    };

    struct B {
      bool operator()() const;  // expected-warning {{function 'test11::(anonymous namespace)::B::operator()' has internal linkage but is not defined}}
      void operator!() const;  // expected-warning {{function 'test11::(anonymous namespace)::B::operator!' has internal linkage but is not defined}}
      bool operator+(const B&) const;  // expected-warning {{function 'test11::(anonymous namespace)::B::operator+' has internal linkage but is not defined}}
      int operator[](int) const;  // expected-warning {{function 'test11::(anonymous namespace)::B::operator[]' has internal linkage but is not defined}}
      const B* operator->() const;  // expected-warning {{function 'test11::(anonymous namespace)::B::operator->' has internal linkage but is not defined}}
      int member;
    };
  }

  void test1(A &a1, A &a2) {
    a1();
    !a1;
    a1 + a2;
    a1[0];
    (void)a1->member;
  }

  void test2(B &b1, B &b2) {
    b1();  // expected-note {{used here}}
    !b1;  // expected-note {{used here}}
    b1 + b2;  // expected-note {{used here}}
    b1[0];  // expected-note {{used here}}
    (void)b1->member;  // expected-note {{used here}}
  }
}

namespace test12 {
  class T1 {}; class T2 {}; class T3 {}; class T4 {}; class T5 {}; class T6 {};
  class T7 {};

  namespace {
    struct Cls {
      virtual void f(int) = 0;
      virtual void f(int, double) = 0;
      void g(int);  // expected-warning {{function 'test12::(anonymous namespace)::Cls::g' has internal linkage but is not defined}}
      void g(int, double);
      virtual operator T1() = 0;
      virtual operator T2() = 0;
      virtual operator T3&() = 0;
      operator T4();  // expected-warning {{function 'test12::(anonymous namespace)::Cls::operator T4' has internal linkage but is not defined}}
      operator T5();  // expected-warning {{function 'test12::(anonymous namespace)::Cls::operator T5' has internal linkage but is not defined}}
      operator T6&();  // expected-warning {{function 'test12::(anonymous namespace)::Cls::operator test12::T6 &' has internal linkage but is not defined}}
    };

    struct Cls2 {
      Cls2(T7);  // expected-warning {{function 'test12::(anonymous namespace)::Cls2::Cls2' has internal linkage but is not defined}}
    };
  }

  void test(Cls &c) {
    c.f(7);
    c.g(7);  // expected-note {{used here}}
    (void)static_cast<T1>(c);
    T2 t2 = c;
    T3 &t3 = c;
    (void)static_cast<T4>(c); // expected-note {{used here}}
    T5 t5 = c;  // expected-note {{used here}}
    T6 &t6 = c;  // expected-note {{used here}}

    Cls2 obj1((T7()));  // expected-note {{used here}}
  }
}

namespace test13 {
  namespace {
    struct X {
      virtual void f() { }
    };

    struct Y : public X {
      virtual void f() = 0;

      virtual void g() {
        X::f();
      }
    };
  }
}

namespace test14 {
  extern "C" const int foo;

  int f() {
    return foo;
  }
}
