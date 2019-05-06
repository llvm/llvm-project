// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -verify -std=c++11 -Wall %s

struct Bitfield {
  int n : 3 = 7; // expected-warning {{C++2a extension}} expected-warning {{changes value from 7 to -1}}
};

int a;
class NoWarning {
  int &n = a;
public:
  int &GetN() { return n; }
};

bool b();
int k;
struct Recurse { // expected-error {{initializer for 'n' needed}}
  int &n = // expected-note {{declared here}}
      b() ?
      Recurse().n : // expected-note {{in evaluation of exception spec}}
      k;
};

struct UnknownBound {
  int as[] = { 1, 2, 3 }; // expected-error {{array bound cannot be deduced from an in-class initializer}}
  int bs[4] = { 4, 5, 6, 7 };
  int cs[] = { 8, 9, 10 }; // expected-error {{array bound cannot be deduced from an in-class initializer}}
};

template<int n> struct T { static const int B; };
template<> struct T<2> { template<int C, int D> using B = int; };
const int C = 0, D = 0;
struct S {
  int as[] = { decltype(x)::B<C, D>(0) }; // expected-error {{array bound cannot be deduced from an in-class initializer}}
  T<sizeof(as) / sizeof(int)> x;
  // test that we handle invalid array bound deductions without crashing when the declarator name is itself invalid
  operator int[](){}; // expected-error {{'operator int' cannot be the name of a variable or data member}} \
                      // expected-error {{array bound cannot be deduced from an in-class initializer}}
};

struct ThrowCtor { ThrowCtor(int) noexcept(false); };
struct NoThrowCtor { NoThrowCtor(int) noexcept(true); };

struct Throw { ThrowCtor tc = 42; };
struct NoThrow { NoThrowCtor tc = 42; };

static_assert(!noexcept(Throw()), "incorrect exception specification");
static_assert(noexcept(NoThrow()), "incorrect exception specification");

struct CheckExcSpec {
  CheckExcSpec() noexcept(true) = default;
  int n = 0;
};
struct CheckExcSpecFail {
  CheckExcSpecFail() noexcept(true) = default; // ok, but calls terminate() on exception
  ThrowCtor tc = 123;
};

struct TypedefInit {
  typedef int A = 0; // expected-error {{illegal initializer}}
};

// PR10578 / <rdar://problem/9877267>
namespace PR10578 {
  template<typename T>
  struct X { 
    X() {
      T* x = 1; // expected-error{{cannot initialize a variable of type 'int *' with an rvalue of type 'int'}}
    }
  };

  struct Y : X<int> {
    Y();
  };

  Y::Y() try { // expected-note{{in instantiation of member function 'PR10578::X<int>::X' requested here}}
  } catch(...) {
  }
}

namespace PR14838 {
  struct base { ~base() {} };
  class function : base {
    ~function() {} // expected-note {{implicitly declared private here}}
  public:
    function(...) {}
  };
  struct thing {};
  struct another {
    another() : r(thing()) {} // expected-error {{binds to a temporary object}}
    // expected-error@-1 {{temporary of type 'PR14838::function' has private destructor}}
    const function &r; // expected-note {{reference member declared here}}
  } af;
}

namespace rdar14084171 {
  struct Point { // expected-note 3 {{candidate constructor}}
    double x;
    double y;
  };
  struct Sprite {
    Point location = Point(0,0); // expected-error {{no matching constructor for initialization of 'rdar14084171::Point'}}
  };
  void f(Sprite& x) { x = x; } // expected-warning {{explicitly assigning value of variable}}
}

namespace PR18560 {
  struct X { int m; };

  template<typename T = X,
           typename U = decltype(T::m)>
  int f();

  struct Y { int b = f(); };
}

namespace template_valid {
// Valid, we shouldn't build a CXXDefaultInitExpr until A's ctor definition.
struct A {
  A();
  template <typename T>
  struct B { int m1 = sizeof(A) + sizeof(T); };
  B<int> m2;
};
A::A() {}
}

namespace template_default_ctor {
struct A {
  template <typename T>
  struct B { // expected-error {{initializer for 'm1' needed}}
    int m1 = 0; // expected-note {{declared here}}
  };
  enum { NOE = noexcept(B<int>()) }; // expected-note {{in evaluation of exception spec}}
};
}

namespace default_ctor {
struct A {
  struct B { // expected-error {{initializer for 'm1' needed}}
    int m1 = 0; // expected-note {{declared here}}
  };
  enum { NOE = noexcept(B()) }; // expected-note {{in evaluation of exception spec}}
};
}

namespace member_template {
struct A {
  template <typename T>
  struct B {
    struct C { // expected-error {{initializer for 'm1' needed}}
      int m1 = 0; // expected-note {{declared here}}
    };
    template <typename U>
    struct D { // expected-error {{initializer for 'm1' needed}}
      int m1 = 0; // expected-note {{declared here}}
    };
  };
  enum {
    NOE1 = noexcept(B<int>::C()), // expected-note {{in evaluation of exception spec}}
    NOE2 = noexcept(B<int>::D<int>()) // expected-note {{in evaluation of exception spec}}
  };
};
}

namespace explicit_instantiation {
template<typename T> struct X {
  X(); // expected-note {{in instantiation of default member initializer 'explicit_instantiation::X<float>::n' requested here}}
  int n = T::error; // expected-error {{type 'float' cannot be used prior to '::' because it has no members}}
};
template struct X<int>; // ok
template<typename T> X<T>::X() {}
template struct X<float>; // expected-note {{in instantiation of member function 'explicit_instantiation::X<float>::X' requested here}}
}

namespace local_class {
template<typename T> void f() {
  struct X { // expected-note {{in instantiation of default member initializer 'local_class::f()::X::n' requested here}}
    int n = T::error; // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}
  };
}
void g() { f<int>(); } // expected-note {{in instantiation of function template specialization 'local_class::f<int>' requested here}}
}

namespace PR22056 {
template <int N>
struct S {
  int x[3] = {[N] = 3};
};
}

namespace PR28060 {
template <class T>
void foo(T v) {
  struct s {
    T *s = 0;
  };
}
template void foo(int);
}
