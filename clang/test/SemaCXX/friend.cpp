// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++14

friend class A; // expected-error {{'friend' used outside of class}}
void f() { friend class A; } // expected-error {{'friend' used outside of class}}
class C { friend class A; };
class D { void f() { friend class A; } }; // expected-error {{'friend' used outside of class}}

// PR5760
namespace test0 {
  namespace ns {
    void f(int);
  }

  struct A {
    friend void ns::f(int a);
  };
}

// Test derived from LLVM's Registry.h
namespace test1 {
  template <class T> struct Outer {
    void foo(T);
    struct Inner {
      friend void Outer::foo(T);
    };
  };

  void test() {
    (void) Outer<int>::Inner();
  }
}

// PR5476
namespace test2 {
  namespace foo {
    void Func(int x);
  }

  class Bar {
    friend void ::test2::foo::Func(int x);
  };
}

// PR5134
namespace test3 {
  class Foo {
    friend const int getInt(int inInt = 0) {}

  };
}

namespace test4 {
  class T4A {
    friend class T4B;
  
  public:
    T4A(class T4B *);

  protected:
    T4B *mB;          // error here
  };
 
  class T4B {};
}

namespace rdar8529993 {
struct A { ~A(); };

struct B : A
{
  template<int> friend A::~A(); // expected-error {{destructor cannot be declared as a template}}
};
}

// PR7915
namespace test5 {
  struct A;
  struct A1 { friend void A(); };

  struct B { friend void B(); };
}

// PR8479
namespace test6_1 {
  class A {
   public:
   private:
    friend class vectorA;
    A() {}
  };
  class vectorA {
   public:
    vectorA(int i, const A& t = A()) {}
  };
  void f() {
    vectorA v(1);
  }
}
namespace test6_2 {
  template<class T>
  class vector {
   public:
    vector(int i, const T& t = T()) {}
  };
  class A {
   public:
   private:
    friend class vector<A>;
    A() {}
  };
  void f() {
    vector<A> v(1);
  }
}
namespace test6_3 {
  template<class T>
  class vector {
   public:
    vector(int i) {}
    void f(const T& t = T()) {}
  };
  class A {
   public:
   private:
    friend void vector<A>::f(const A&);
    A() {}
  };
  void f() {
    vector<A> v(1);
    v.f();
  }
}

namespace test7 {
  extern "C" {
    class X {
      friend int test7_f() { return 42; }
    };
  }
}

// PR15485
namespace test8 {
  namespace ns1 {
    namespace ns2 {
      template<class T> void f(T t); // expected-note {{target of using declaration}}
    }
    using ns2::f; // expected-note {{using declaration}}
  }
  struct A { void f(); }; // expected-note 2{{target of using declaration}}
  struct B : public A { using A::f; }; // expected-note {{using declaration}}
  template<typename T> struct C : A { using A::f; }; // expected-note {{using declaration}}
  struct X {
    template<class T> friend void ns1::f(T t); // expected-error {{cannot befriend target of using declaration}}
    friend void B::f(); // expected-error {{cannot befriend target of using declaration}}
    friend void C<int>::f(); // expected-error {{cannot befriend target of using declaration}}
  };
}

// PR16423
namespace test9 {
  class C {
  };
  struct A {
    friend void C::f(int, int, int) {}  // expected-error {{friend declaration of 'f' does not match any declaration in 'test9::C'}}
  };
}

namespace test10 {
  struct X {};
  extern void f10_a();
  extern void f10_a(X);
  struct A {
    friend void f10_a();
    friend void f10_b();
    friend void f10_c();
    friend void f10_d();
    friend void f10_a(X);
    friend void f10_b(X);
    friend void f10_c(X);
    friend void f10_d(X);
  };
  extern void f10_b();
  extern void f10_b(X);
  struct B {
    friend void f10_a();
    friend void f10_b();
    friend void f10_c();
    friend void f10_d();
    friend void f10_a(X);
    friend void f10_b(X);
    friend void f10_c(X);
    friend void f10_d(X);
  };
  extern void f10_c();
  extern void f10_c(X);

  // FIXME: Give a better diagnostic for the case where a function exists but is
  // not visible.
  void g(X x) {
    f10_a();
    f10_b();
    f10_c();
    f10_d(); // expected-error {{undeclared identifier}}

    ::test10::f10_a();
    ::test10::f10_b();
    ::test10::f10_c();
    ::test10::f10_d(); // expected-error {{no member named 'f10_d'}}

    f10_a(x);
    f10_b(x);
    f10_c(x);
    f10_d(x); // PR16597: expected-error {{undeclared identifier}}

    ::test10::f10_a(x);
    ::test10::f10_b(x);
    ::test10::f10_c(x);
    ::test10::f10_d(x); // expected-error {{no type named 'f10_d'}}
  }

  struct Y : X {
    friend void f10_d();
    friend void f10_d(X);
  };

  struct Z {
    operator X();
    friend void f10_d();
    friend void f10_d(X);
  };

  struct W {
    friend void f10_d(W);
  };

  void g(X x, Y y, Z z) {
    f10_d(); // expected-error {{undeclared identifier}}
    ::test10::f10_d(); // expected-error {{no member named 'f10_d'}}

    // f10_d is visible to ADL in the second and third cases.
    f10_d(x); // expected-error {{undeclared identifier}}
    f10_d(y);
    f10_d(z);

    // No ADL here.
    ::test10::f10_d(x); // expected-error {{no type named 'f10_d'}}
    ::test10::f10_d(y); // expected-error {{no type named 'f10_d'}}
    ::test10::f10_d(z); // expected-error {{no type named 'f10_d'}}
  }

  void local_externs(W w, X x, Y y) {
    extern void f10_d(); // expected-note {{candidate}}
    extern void f10_d(X); // expected-note {{candidate}}
    f10_d();
    f10_d(x);
    f10_d(y);
    f10_d(w); // expected-error {{no matching}}
    {
      int f10_d;
      f10_d(); // expected-error {{not a function}}
      f10_d(x); // expected-error {{not a function}}
      f10_d(y); // expected-error {{not a function}}
    }
  }

  void i(X x, Y y) {
    f10_d(); // expected-error {{undeclared identifier}}
    f10_d(x); // expected-error {{undeclared identifier}}
    f10_d(y);
  }

  struct C {
    friend void f10_d();
    friend void f10_d(X);
  };

  void j(X x, Y y) {
    f10_d(); // expected-error {{undeclared identifier}}
    f10_d(x); // expected-error {{undeclared identifier}}
    f10_d(y);
  }

  extern void f10_d();
  extern void f10_d(X);
  void k(X x, Y y, Z z) {
    // All OK now.
    f10_d();
    f10_d(x);
    ::test10::f10_d();
    ::test10::f10_d(x);
    ::test10::f10_d(y);
    ::test10::f10_d(z);
  }
}

namespace test11 {
  class __attribute__((visibility("hidden"))) B;

  class A {
    friend class __attribute__((visibility("hidden"), noreturn)) B; // expected-warning {{'noreturn' attribute only applies to functions and methods}}
  };
}

namespace pr21851 {
// PR21851 was a problem where we assumed that when the friend function redecl
// lookup found a C++ method, it would necessarily have a qualifier. Below we
// have some test cases where unqualified lookup finds C++ methods without using
// qualifiers. Unfortunately, we can't exercise the case of an access check
// failure because nested classes always have access to the members of outer
// classes.

void friend_own_method() {
  class A {
    void m() {}
    friend void m();
  };
}

void friend_enclosing_method() {
  class A;
  class C {
    int p;
    friend class A;
  };
  class A {
    void enclosing_friend() {
      (void)b->p;
      (void)c->p;
    }
    class B {
      void b(A *a) {
        (void)a->c->p;
      }
      int p;
      friend void enclosing_friend();
    };
    B *b;
    C *c;
  };
}

static auto friend_file_func() {
  extern void file_scope_friend();
  class A {
    int p;
    friend void file_scope_friend();
  };
  return A();
}

void file_scope_friend() {
  auto a = friend_file_func();
  (void)a.p;
}
}

template<typename T>
struct X_pr6954 {
  operator int();
  friend void f_pr6954(int x);
};

int array0_pr6954[sizeof(X_pr6954<int>)];
int array1_pr6954[sizeof(X_pr6954<float>)];

void g_pr6954() {
  f_pr6954(5); // expected-error{{undeclared identifier 'f_pr6954'}}
}

namespace tag_redecl {
  namespace N {
    struct X *p;
    namespace {
      class K {
        friend struct X;
      };
    }
  }
  namespace N {
    struct X;
    X *q = p;
  }
}

namespace default_arg {
  void f();
  void f(void*); // expected-note {{previous}}
  struct X {
    friend void f(int a, int b = 0) {}
    friend void f(void *p = 0) {} // expected-error {{must be the only}}
  };
}

namespace PR33222 {
  int f();
  template<typename T> struct X {
    friend T f();
  };
  X<int> xi;

  int g(); // expected-note {{previous}}
  template<typename T> struct Y {
    friend T g(); // expected-error {{return type}}
  };
  Y<float> yf; // expected-note {{instantiation}}

  int h(); // expected-note {{previous}}
  template<typename T> struct Z {
    friend T h(); // expected-error {{return type}}
  };
  Z<int> zi;
  Z<float> zf; // expected-note {{instantiation}}
}

namespace qualified_friend_no_match {
  void f(int); // expected-note {{type mismatch at 1st parameter}}
  template<typename T> void f(T*); // expected-note {{could not match 'T *' against 'double'}}
  struct X {
    friend void qualified_friend_no_match::f(double); // expected-error {{friend declaration of 'f' does not match any declaration in namespace 'qualified_friend_no_match'}}
    friend void qualified_friend_no_match::g(); // expected-error {{friend declaration of 'g' does not match any declaration in namespace 'qualified_friend_no_match'}}
  };

  struct Y {
    void f(int); // expected-note {{type mismatch at 1st parameter}}
    template<typename T> void f(T*); // expected-note {{could not match 'T *' against 'double'}}
  };
  struct Z {
    friend void Y::f(double); // expected-error {{friend declaration of 'f' does not match any declaration in 'qualified_friend_no_match::Y'}}
  };
}
