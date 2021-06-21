// RUN: %clang_cc1 -std=c++2b -fsyntax-only -verify=expected,cxx20_2b,cxx2b    %s
// RUN: %clang_cc1 -std=c++2b -fsyntax-only -verify=expected,cxx20_2b,cxx2b    %s -fdelayed-template-parsing -DDELAYED_TEMPLATE_PARSING

// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,cxx14_20,cxx20_2b %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,cxx14_20,cxx20_2b %s -fdelayed-template-parsing -DDELAYED_TEMPLATE_PARSING

// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify=expected,cxx14_20,cxx14    %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify=expected,cxx14_20,cxx14    %s -fdelayed-template-parsing -DDELAYED_TEMPLATE_PARSING

auto f(); // expected-note {{previous}}
int f(); // expected-error {{differ only in their return type}}

auto &g();
auto g() -> auto &;

auto h() -> auto *;
auto *h();

struct Conv1 {
  operator auto(); // expected-note {{declared here}}
} conv1;
int conv1a = conv1; // expected-error {{function 'operator auto' with deduced return type cannot be used before it is defined}}
// expected-error@-1 {{no viable conversion}}
Conv1::operator auto() { return 123; }
int conv1b = conv1;
int conv1c = conv1.operator auto();
int conv1d = conv1.operator int(); // expected-error {{no member named 'operator int'}}

struct Conv2 {
  operator auto() { return 0; }  // expected-note {{previous}}
  operator auto() { return 0.; } // expected-error {{cannot be redeclared}}
};

struct Conv3 {
  operator auto() { int *p = nullptr; return p; }  // expected-note {{candidate}}
  operator auto*() { int *p = nullptr; return p; } // expected-note {{candidate}}
} conv3;
int *conv3a = conv3; // expected-error {{ambiguous}}
int *conv3b = conv3.operator auto();
int *conv3c = conv3.operator auto*();

template<typename T>
struct Conv4 {
  operator auto() { return T(); }
};
Conv4<int> conv4int;
int conv4a = conv4int;
int conv4b = conv4int.operator auto();

auto a();
auto a() { return 0; }
using T = decltype(a());
using T = int;
auto a(); // expected-note {{previous}}
using T = decltype(a());
auto *a(); // expected-error {{differ only in their return type}}

auto b(bool k) {
  if (k)
    return "hello";
  return "goodbye";
}

// Allow 'operator auto' to call only the explicit operator auto.
struct BothOps {
  template <typename T> operator T();
  template <typename T> operator T *();
  operator auto() { return 0; }
  operator auto *() { return this; }
};
struct JustTemplateOp {
  template <typename T> operator T();
  template <typename T> operator T *();
};

auto c() {
  BothOps().operator auto(); // ok
  BothOps().operator auto *(); // ok
  JustTemplateOp().operator auto(); // expected-error {{no member named 'operator auto' in 'JustTemplateOp'}}
  JustTemplateOp().operator auto *(); // expected-error {{no member named 'operator auto *' in 'JustTemplateOp'}}
}

auto *ptr_1() {
  return 100; // expected-error {{cannot deduce return type 'auto *' from returned value of type 'int'}}
}

const auto &ref_1() {
  return 0; // expected-warning {{returning reference to local temporary}}
}

auto init_list() {
  return { 1, 2, 3 }; // expected-error {{cannot deduce return type from initializer list}}
}

auto fwd_decl(); // expected-note 2{{here}}

int n = fwd_decl(); // expected-error {{function 'fwd_decl' with deduced return type cannot be used before it is defined}}
int k = sizeof(fwd_decl()); // expected-error {{used before it is defined}}

auto fac(int n) {
  if (n <= 2)
    return n;
  return n * fac(n-1); // ok
}

auto fac_2(int n) { // expected-note {{declared here}}
  if (n > 2)
    return n * fac_2(n-1); // expected-error {{cannot be used before it is defined}}
  return n;
}

auto void_ret() {}
using Void = void;
using Void = decltype(void_ret());

auto &void_ret_2() {} // expected-error {{cannot deduce return type 'auto &' for function with no return statements}}
const auto void_ret_3() {} // ok, return type 'const void' is adjusted to 'void'

const auto void_ret_4() {
  if (false)
    return void();
  if (false)
    return;
  return 0; // expected-error {{'auto' in return type deduced as 'int' here but deduced as 'void' in earlier return statement}}
}

namespace Templates {
  template<typename T> auto f1() {
    return T() + 1;
  }
  template<typename T> auto &f2(T &&v) { return v; }
  // cxx2b-error@-1 {{non-const lvalue reference to type 'int' cannot bind to a temporary of type 'int'}}
  // cxx2b-error@-2 {{non-const lvalue reference to type 'double' cannot bind to a temporary of type 'double'}}
  // cxx2b-note@-3  {{candidate template ignored: substitution failure [with T = double]}}
  int a = f1<int>();
  const int &b = f2(0); // cxx2b-note {{in instantiation of function template specialization 'Templates::f2<int>' requested here}}
  double d;
  float &c = f2(0.0); // expected-error {{non-const lvalue reference to type 'float' cannot bind to a value of unrelated type 'double'}}
  // cxx2b-note@-1 {{in instantiation of function template specialization 'Templates::f2<double>' requested here}}

  template<typename T> auto fwd_decl(); // expected-note {{declared here}}
  int e = fwd_decl<int>(); // expected-error {{cannot be used before it is defined}}
  template<typename T> auto fwd_decl() { return 0; }
  int f = fwd_decl<int>();
  template <typename T>
  auto fwd_decl(); // expected-note {{candidate template ignored: could not match 'auto ()' against 'int ()'}}
  int g = fwd_decl<char>();

  auto (*p)() = f1; // expected-error {{incompatible initializer}}
  auto (*q)() = f1<int>; // ok

  typedef decltype(f2(1.2)) dbl; // cxx14_20-note {{previous}}
  // cxx2b-error@-1 {{no matching function for call to 'f2'}}
  typedef float dbl; // cxx14_20-error {{typedef redefinition with different types ('float' vs 'decltype(f2(1.2))' (aka 'double &'))}}

  extern template auto fwd_decl<double>();
  int k1 = fwd_decl<double>();
  extern template int fwd_decl<char>(); // expected-error {{does not refer to a function template}}
  int k2 = fwd_decl<char>();

  template <typename T> auto instantiate() { T::error; } // expected-error {{has no members}} \
    // expected-note {{candidate template ignored: could not match 'auto ()' against 'void ()'}}
  extern template auto instantiate<int>(); // ok
  int k = instantiate<int>(); // expected-note {{in instantiation of}}
  template<> auto instantiate<char>() {} // ok
  template<> void instantiate<double>() {} // expected-error {{no function template matches}}

  template<typename T> auto arg_single() { return 0; }
  template<typename T> auto arg_multi() { return 0l; }
  template<typename T> auto arg_multi(int) { return "bad"; }
  template<typename T> struct Outer {
    static auto arg_single() { return 0.f; }
    static auto arg_multi() { return 0.; }
    static auto arg_multi(int) { return "bad"; }
  };
  template<typename T> T &take_fn(T (*p)());

  int &check1 = take_fn(arg_single); // expected-error {{no matching}} expected-note@-2 {{couldn't infer}}
  int &check2 = take_fn(arg_single<int>);
  int &check3 = take_fn<int>(arg_single); // expected-error {{no matching}} expected-note@-4{{no overload of 'arg_single'}}
  int &check4 = take_fn<int>(arg_single<int>);
  long &check5 = take_fn(arg_multi); // expected-error {{no matching}} expected-note@-6 {{couldn't infer}}
  long &check6 = take_fn(arg_multi<int>);
  long &check7 = take_fn<long>(arg_multi); // expected-error {{no matching}} expected-note@-8{{no overload of 'arg_multi'}}
  long &check8 = take_fn<long>(arg_multi<int>);

  float &mem_check1 = take_fn(Outer<int>::arg_single);
  float &mem_check2 = take_fn<float>(Outer<char>::arg_single);
  double &mem_check3 = take_fn(Outer<long>::arg_multi);
  double &mem_check4 = take_fn<double>(Outer<double>::arg_multi);

  namespace Deduce1 {
  template <typename T> auto f() { return 0; } // expected-note {{couldn't infer template argument 'T'}}
    template<typename T> void g(T(*)()); // expected-note 2{{candidate}}
    void h() {
      auto p = f<int>;
      auto (*q)() = f<int>;
      int (*r)() = f; // expected-error {{does not match}}
      g(f<int>);
      g<int>(f); // expected-error {{no matching function}}
      g(f); // expected-error {{no matching function}}
    }
  }

  namespace Deduce2 {
  template <typename T> auto f(int) { return 0; } // expected-note {{couldn't infer template argument 'T'}}
    template<typename T> void g(T(*)(int)); // expected-note 2{{candidate}}
    void h() {
      auto p = f<int>;
      auto (*q)(int) = f<int>;
      int (*r)(int) = f; // expected-error {{does not match}}
      g(f<int>);
      g<int>(f); // expected-error {{no matching function}}
      g(f); // expected-error {{no matching function}}
    }
  }

  namespace Deduce3 {
    template<typename T> auto f(T) { return 0; }
    template<typename T> void g(T(*)(int)); // expected-note {{couldn't infer}}
    void h() {
      auto p = f<int>;
      auto (*q)(int) = f<int>;
      int (*r)(int) = f; // ok
      g(f<int>);
      g<int>(f); // ok
      g(f); // expected-error {{no matching function}}
    }
  }

  namespace DeduceInDeducedReturnType {
    template<typename T, typename U> auto f() -> auto (T::*)(U) {
      int (T::*result)(U) = nullptr;
      return result;
    }
    struct S {};
    int (S::*(*p)())(double) = f;
    int (S::*(*q)())(double) = f<S, double>;
  }
}

auto fwd_decl_using();
namespace N { using ::fwd_decl_using; }
auto fwd_decl_using() { return 0; }
namespace N { int k = N::fwd_decl_using(); }

namespace OverloadResolutionNonTemplate {
  auto f();
  auto f(int); // expected-note {{here}}

  int &g(int (*f)()); // expected-note {{not viable: no overload of 'f' matching 'int (*)()'}}
  char &g(int (*f)(int)); // expected-note {{not viable: no overload of 'f' matching 'int (*)(int)'}}

  int a = g(f); // expected-error {{no matching function}}

  auto f() { return 0; }

  // FIXME: It's not completely clear whether this should be ill-formed.
  int &b = g(f); // expected-error {{used before it is defined}}

  auto f(int) { return 0.0; }

  int &c = g(f); // ok
}

namespace OverloadResolutionTemplate {
  auto f();
  template<typename T> auto f(T);

  int &g(int (*f)()); // expected-note {{not viable: no overload of 'f' matching 'int (*)()'}} expected-note {{candidate}}
  char &g(int (*f)(int)); // expected-note {{not viable: no overload of 'f' matching 'int (*)(int)'}} expected-note {{candidate}}

  int a = g(f); // expected-error {{no matching function}}

  auto f() { return 0; }

  int &b = g(f); // ok (presumably), due to deduction failure forming type of 'f<int>'

  template<typename T> auto f(T) { return 0; }

  int &c = g(f); // expected-error {{ambiguous}}
}

namespace DefaultedMethods {
  struct A {
    auto operator=(const A&) = default; // expected-error {{must return 'DefaultedMethods::A &'}}
    A &operator=(A&&); // expected-note {{previous}}
  };
  auto A::operator=(A&&) = default; // expected-error {{return type of out-of-line definition of 'DefaultedMethods::A::operator=' differs from that in the declaration}}
}

namespace Constexpr {
  constexpr auto f1(int n) { return n; }
  template<typename T> struct X { constexpr auto f() {} }; // PR18746
  template<typename T> struct Y { constexpr T f() {} }; // expected-note {{control reached end of constexpr function}}
  void f() {
    X<int>().f();
    Y<void>().f();
    constexpr int q = Y<int>().f(); // expected-error {{must be initialized by a constant expression}} expected-note {{in call to '&Y<int>()->f()'}}
  }
  struct NonLiteral { ~NonLiteral(); } nl; // cxx14-note {{user-provided destructor}}
  // cxx20_2b-note@-1 {{'NonLiteral' is not literal because its destructor is not constexpr}}
  constexpr auto f2(int n) { return nl; } // expected-error {{return type 'Constexpr::NonLiteral' is not a literal type}}
}

// It's not really clear whether these are valid, but this matches g++.
using size_t = decltype(sizeof(0));
auto operator new(size_t n, const char*); // expected-error {{must return type 'void *'}}
auto operator delete(void *, const char*); // expected-error {{must return type 'void'}}

namespace Virtual {
  struct S {
    virtual auto f() { return 0; } // expected-error {{function with deduced return type cannot be virtual}} expected-note {{here}}
  };
  // Allow 'auto' anyway for error recovery.
  struct T : S {
    int f();
  };
  struct U : S {
    auto f(); // expected-error {{different return}}
  };

  // And here's why...
  struct V { virtual auto f(); }; // expected-error {{cannot be virtual}}
  struct W : V { virtual auto f(); }; // expected-error {{cannot be virtual}}
  auto V::f() { return 0; } // in tu1.cpp
  auto W::f() { return 0.0; } // in tu2.cpp
  W w;
  int k1 = w.f();
  int k2 = ((V&)w).f();
}

namespace std_examples {

namespace NoReturn {
  auto f() {}
  void (*p)() = &f;

  auto f(); // ok

  auto *g() {} // expected-error {{cannot deduce return type 'auto *' for function with no return statements}}

  auto h() = delete; // expected-note {{explicitly deleted}}
  auto x = h(); // expected-error {{call to deleted}}
}

namespace UseBeforeComplete {
  auto n = n; // expected-error {{variable 'n' declared with deduced type 'auto' cannot appear in its own initializer}}
  auto f(); // expected-note {{declared here}}
  void g() { &f; } // expected-error {{function 'f' with deduced return type cannot be used before it is defined}}
  auto sum(int i) {
    if (i == 1)
      return i;
    else
      return sum(i - 1) + i;
  }
}

namespace Redecl {
  auto f();
  auto f() { return 42; }
  auto f(); // expected-note 2{{previous}}
  int f(); // expected-error {{functions that differ only in their return type cannot be overloaded}}
  decltype(auto) f(); // expected-error {{cannot be overloaded}}

  template <typename T> auto g(T t) { return t; } // expected-note {{candidate}} \
                                                  // expected-note {{candidate function [with T = int]}}
  template auto g(int);
  template char g(char); // expected-error {{does not refer to a function}}
  template<> auto g(double);

  template<typename T> T g(T t) { return t; } // expected-note {{candidate}}
  template char g(char);
  template auto g(float);

  void h() { return g(42); } // expected-error {{ambiguous}}
}

namespace ExplicitInstantiationDecl {
  template<typename T> auto f(T t) { return t; }
  extern template auto f(int);
  int (*p)(int) = f;
}
namespace MemberTemplatesWithDeduction {
  struct M {
    template<class T> auto foo(T t) { return t; }
    template<class T> auto operator()(T t) const { return t; }
    template<class T> static __attribute__((unused)) int static_foo(T) {
      return 5;
    }
    template<class T> operator T() { return T{}; }
    operator auto() { return &static_foo<int>; }
  };
  struct N : M {
    using M::foo;
    using M::operator();
    using M::static_foo;
    using M::operator auto;
  };

  template <class T> int test() {
    int i = T{}.foo(3);
    T m = T{}.foo(M{});
    int j = T{}(3);
    M m2 = M{}(M{});
    int k = T{}.static_foo(4);
    int l = T::static_foo(5);
    int l2 = T{};
    struct X { };
    X x = T{};
    return 0;
  }
  int Minst = test<M>();
  int Ninst = test<N>();

}
}

// We resolve a wording bug here: 'decltype(auto)' should not be modeled as a
// decltype-specifier, just as a simple-type-specifier. All the extra places
// where a decltype-specifier can appear make no sense for 'decltype(auto)'.
namespace DecltypeAutoShouldNotBeADecltypeSpecifier {
  namespace NNS {
    int n;
    decltype(auto) i();
    decltype(n) j();
    struct X {
      friend decltype(auto) ::DecltypeAutoShouldNotBeADecltypeSpecifier::NNS::i();
      friend decltype(n) ::DecltypeAutoShouldNotBeADecltypeSpecifier::NNS::j(); // expected-error {{not a class}}
    };
  }

  namespace Dtor {
    struct A {};
    void f(A a) { a.~decltype(auto)(); } // expected-error {{'decltype(auto)' not allowed here}}
    void g(int i) { i.~decltype(auto)(); } // expected-error {{'decltype(auto)' not allowed here}}
  }

  namespace BaseClass {
    struct A : decltype(auto) {}; // expected-error {{'decltype(auto)' not allowed here}}
    struct B {
      B() : decltype(auto)() {} // expected-error {{'decltype(auto)' not allowed here}}
    };
  }
}

namespace CurrentInstantiation {
  // PR16875
  template<typename T> struct S {
    auto f() { return T(); }
    int g() { return f(); }
    auto h(bool b) {
      if (b)
        return T();
      return h(true);
    }
  };
  int k1 = S<int>().g();
  int k2 = S<int>().h(false);

  template<typename T> struct U {
 #ifndef DELAYED_TEMPLATE_PARSING
    auto f(); // expected-note {{here}}
    int g() { return f(); } // expected-error {{cannot be used before it is defined}}
 #else
    auto f();
    int g() { return f(); }
 #endif
  };
 #ifndef DELAYED_TEMPLATE_PARSING
  template int U<int>::g(); // expected-note {{in instantiation of}}
 #else
  template int U<int>::g();
 #endif
  template<typename T> auto U<T>::f() { return T(); }
  template int U<short>::g(); // ok
}

namespace WithDefaultArgs {
  template<typename U> struct A {
    template<typename T = U> friend auto f(A) { return []{}; }
  };
  template<typename T> void f();
  using T = decltype(f(A<int>()));
  using T = decltype(f<int>(A<int>()));
}

namespace MultilevelDeduction {

auto F() -> auto* { return (int*)0; }

auto (*G())() -> int* { return F; }

auto run = G();

namespace Templated {
template<class T>
auto F(T t) -> auto* { return (T*)0; }

template<class T>
auto (*G(T t))(T) -> T* { return &F<T>; }


template<class T>
auto (*G2(T t))(T) -> auto* { return &F<T>; }

auto run_int = G(1);
auto run_char = G2('a');

}
}

namespace rnk {
extern "C" int puts(const char *s);
template <typename T>
auto foo(T x) -> decltype(x) {
#ifdef DELAYED_TEMPLATE_PARSING
  ::rnk::bar();
#endif
  return x;
}
void bar() { puts("bar"); }
int main() { return foo(0); }

}

namespace OverloadedOperators {
  template<typename T> struct A {
    auto operator()() { return T{}; }
    auto operator[](int) { return T{}; }
    auto operator+(int) { return T{}; }
    auto operator+() { return T{}; }
    friend auto operator-(A) { return T{}; }
    friend auto operator-(A, A) { return T{}; }
  };
  void f(A<int> a) {
    int b = a();
    int c = a[0];
    int d = a + 0;
    int e = +a;
    int f = -a;
    int g = a - a;
  }
}

namespace TrailingReturnTypeForConversionOperator {
  struct X {
    operator auto() -> int { return 0; } // expected-error {{cannot specify any part of a return type in the declaration of a conversion function; put the complete type after 'operator'}}
  } x;
  int k = x.operator auto();

  struct Y {
    operator auto() -> int & { // expected-error {{cannot specify}}
      return 0; // expected-error {{cannot bind to}}
    }
  };
};

namespace PR24989 {
  auto x = [](auto){};
  using T = decltype(x);
  void (T::*p)(int) const = &T::operator();
}

void forinit_decltypeauto() {
  for (decltype(auto) forinit_decltypeauto_inner();;) {} // expected-warning {{interpreted as a function}} expected-note {{replace}}
}

namespace PR33222 {
  auto f1();
  auto f2();

  template<typename T> decltype(auto) g0(T x) { return x.n; }
  template<typename T> decltype(auto) g1(T);
  template<typename T> decltype(auto) g2(T);

  struct X {
    static auto f1();
    static auto f2();

    template<typename T> static decltype(auto) g0(T x) { return x.n; }
    template<typename T> static decltype(auto) g1(T);
    template<typename T> static decltype(auto) g2(T);
  };

  template<typename U> class A {
    friend auto f1();
    friend auto f2();

    friend decltype(auto) g0<>(A);
    template<typename T> friend decltype(auto) g1(T);
    template<typename T> friend decltype(auto) g2(T);

    friend auto X::f1();
    friend auto X::f2();

    // FIXME (PR38882): 'A' names the class template not the injected-class-name here!
    friend decltype(auto) X::g0<>(A<U>);
    // FIXME (PR38882): ::T hides the template parameter if both are named T here!
    template<typename T_> friend decltype(auto) X::g1(T_);
    template<typename T_> friend decltype(auto) X::g2(T_);

    int n;
  };

  auto f1() { return A<int>().n; }
  template<typename T> decltype(auto) g1(T x) { return A<int>().n; }

  auto X::f1() { return A<int>().n; }
  template<typename T> decltype(auto) X::g1(T x) { return A<int>().n; }

  A<int> ai;
  int k1 = g0(ai);
  int k2 = X::g0(ai);

  int k3 = g1(ai);
  int k4 = X::g1(ai);

  auto f2() { return A<int>().n; }
  template<typename T> decltype(auto) g2(T x) { return A<int>().n; }

  auto X::f2() { return A<int>().n; }
  template<typename T> decltype(auto) X::g2(T x) { return A<int>().n; }

  int k5 = g2(ai);
  int k6 = X::g2(ai);

  template<typename> struct B {
    auto *q() { return (float*)0; } // expected-note 2{{previous}}
  };
  template<> auto *B<char[1]>::q() { return (int*)0; }
  template<> auto B<char[2]>::q() { return (int*)0; } // expected-error {{return type}}
  template<> int B<char[3]>::q() { return 0; } // expected-error {{return type}}
}

namespace PR46637 {
  using A = auto () -> auto; // expected-error {{'auto' not allowed in type alias}}
  using B = auto (*)() -> auto; // expected-error {{'auto' not allowed in type alias}}
  template<auto (*)() -> auto> struct X {}; // cxx14-error {{'auto' not allowed in template parameter until C++17}}
  template<typename T> struct Y { T x; };
  Y<auto() -> auto> y; // expected-error {{'auto' not allowed in template argument}}
}
