// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -verify -std=c++11 -Wall %s

struct Bitfield {
  int n : 3 = 7; // expected-error {{bitfield member cannot have an in-class initializer}}
};

int a;
class NoWarning {
  int &n = a;
public:
  int &GetN() { return n; }
};

bool b();
int k;
struct Recurse {
  int &n = b() ? Recurse().n : k; // expected-error {{defaulted default constructor of 'Recurse' cannot be used by non-static data member initializer which appears before end of class definition}}
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
  CheckExcSpecFail() noexcept(true) = default; // expected-error {{exception specification of explicitly defaulted default constructor does not match the calculated one}}
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
