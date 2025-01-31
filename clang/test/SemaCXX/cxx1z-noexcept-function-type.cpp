// RUN: %clang_cc1 -std=c++14 -verify -fexceptions -fcxx-exceptions %s
// RUN: %clang_cc1 -std=c++17 -verify -fexceptions -fcxx-exceptions %s -Wno-dynamic-exception-spec
// RUN: %clang_cc1 -std=c++14 -verify -fexceptions -fcxx-exceptions -Wno-c++1z-compat-mangling -DNO_COMPAT_MANGLING %s
// RUN: %clang_cc1 -std=c++14 -verify -fexceptions -fcxx-exceptions -Wno-noexcept-type -DNO_COMPAT_MANGLING %s

#if __cplusplus > 201402L

template<typename T> void redecl1() noexcept(noexcept(T())) {} // expected-note {{previous}}
template<typename T> void redecl1() noexcept(noexcept(T())); // ok, same type
template<typename T> void redecl1() noexcept(noexcept(T())) {} // expected-error {{redefinition}}

template<bool A, bool B> void redecl2() noexcept(A); // expected-note {{previous}}
template<bool A, bool B> void redecl2() noexcept(B); // expected-error {{does not match previous}}

// These have the same canonical type, but are still different.
template<typename A, typename B> void redecl3() throw(A); // expected-note {{previous}}
template<typename A, typename B> void redecl3() throw(B); // expected-error {{does not match previous}}

typedef int I;
template<bool B> void redecl4(I) noexcept(B);
template<bool B> void redecl4(I) noexcept(B);

void (*init_with_exact_type_a)(int) noexcept = redecl4<true>;
void (*init_with_mismatched_type_a)(int) = redecl4<true>;
auto deduce_auto_from_noexcept_function_ptr_a = redecl4<true>;
using DeducedType_a = decltype(deduce_auto_from_noexcept_function_ptr_a);
using DeducedType_a = void (*)(int) noexcept;

void (*init_with_exact_type_b)(int) = redecl4<false>;
void (*init_with_mismatched_type_b)(int) noexcept = redecl4<false>; // expected-error {{cannot initialize a variable of type}}
auto deduce_auto_from_noexcept_function_ptr_b = redecl4<false>;
using DeducedType_b = decltype(deduce_auto_from_noexcept_function_ptr_b);
using DeducedType_b = void (*)(int);

static_assert(noexcept(init_with_exact_type_a(0)));
static_assert(noexcept((+init_with_exact_type_a)(0)));
static_assert(!noexcept(init_with_exact_type_b(0)));
static_assert(!noexcept((+init_with_exact_type_b)(0)));

// Don't look through casts, use the direct type of the expression.
// FIXME: static_cast here would be reasonable, but is not currently permitted.
static_assert(noexcept(static_cast<decltype(init_with_exact_type_a)>(init_with_exact_type_b)(0))); // expected-error {{is not allowed}}
static_assert(noexcept(reinterpret_cast<decltype(init_with_exact_type_a)>(init_with_exact_type_b)(0)));
static_assert(!noexcept(static_cast<decltype(init_with_exact_type_b)>(init_with_exact_type_a)(0)));

template<bool B> auto get_fn() noexcept -> void (*)() noexcept(B) {}
static_assert(noexcept(get_fn<true>()()));
static_assert(!noexcept(get_fn<false>()()));

namespace DependentDefaultCtorExceptionSpec {
  template<typename> struct T { static const bool value = true; };

  template<class A> struct map {
    typedef A a;
    map() noexcept(T<a>::value) {}
  };

  template<class B> struct multimap {
    typedef B b;
    multimap() noexcept(T<b>::value) {}
  };

  // Don't crash here.
  struct A { multimap<int> Map; } a;

  static_assert(noexcept(A()));

  template <class> struct NoexceptWithThis {
    int ca;
    template <class T> auto foo(T) noexcept(ca) { return true; }
    // expected-error@-1 {{noexcept specifier argument is not a constant expression}}
    // expected-note@-2 {{in instantiation of exception specification}}
    // expected-note@-3 {{implicit use of 'this' pointer is only allowed within the evaluation of a call to a 'constexpr' member function}}
  };
  struct InstantiateFromAnotherClass {
    template <class B, class T = decltype(static_cast<bool (B::*)(int)>(&B::foo))> // expected-note {{in instantiation of function template specialization}}
    // expected-note@-1 {{template parameter is declared here}}
    InstantiateFromAnotherClass(B *) {} // expected-note {{in instantiation of default argument}}
  };
  NoexceptWithThis<int> f{};
  // Don't crash here.
  InstantiateFromAnotherClass b{&f}; // expected-note {{while substituting deduced template arguments into function template}}
}

#endif

namespace CompatWarning {
  struct X;

  // These cases don't change.
  void f0(void p() throw(int));
  auto f0() -> void (*)() noexcept(false);

  // These cases take an ABI break in C++17 because their parameter / return types change.
  void f1(void p() noexcept);
  void f2(void (*p)() noexcept(true));
  void f3(void (&p)() throw());
  void f4(void (X::*p)() throw());
  auto f5() -> void (*)() throw();
  auto f6() -> void (&)() throw();
  auto f7() -> void (X::*)() throw();
#if __cplusplus <= 201402L && !defined(NO_COMPAT_MANGLING)
  // expected-warning@-8 {{mangled name of 'f1' will change in C++17 due to non-throwing exception specification in function signature}}
  // expected-warning@-8 {{mangled name of 'f2' will change in C++17 due to non-throwing exception specification in function signature}}
  // expected-warning@-8 {{mangled name of 'f3' will change in C++17 due to non-throwing exception specification in function signature}}
  // expected-warning@-8 {{mangled name of 'f4' will change in C++17 due to non-throwing exception specification in function signature}}
  // expected-warning@-8 {{mangled name of 'f5' will change in C++17 due to non-throwing exception specification in function signature}}
  // expected-warning@-8 {{mangled name of 'f6' will change in C++17 due to non-throwing exception specification in function signature}}
  // expected-warning@-8 {{mangled name of 'f7' will change in C++17 due to non-throwing exception specification in function signature}}
#endif

  // An instantiation-dependent exception specification needs to be mangled in
  // all language modes, since it participates in SFINAE.
  template<typename T> void g(void() throw(T)); // expected-note {{substitution failure}}
  template<typename T> void g(...) = delete; // expected-note {{deleted}}
  void test_g() { g<void>(nullptr); } // expected-error {{deleted}}

  // An instantiation-dependent exception specification needs to be mangled in
  // all language modes, since it participates in SFINAE.
  template<typename T> void h(void() noexcept(T())); // expected-note {{substitution failure}}
  template<typename T> void h(...) = delete; // expected-note {{deleted}}
  void test_h() { h<void>(nullptr); } // expected-error {{deleted}}
}

namespace ImplicitExceptionSpec {
  struct S {
    ~S();
    void f(const S &s = S());
  };
  S::~S() {}
}

namespace Builtins {
  // Pick two functions that ought to have the same noexceptness.
  extern "C" int strcmp(const char *, const char *);
  extern "C" int strncmp(const char *, const char *, decltype(sizeof(0))) noexcept;

  // Check we recognized both as builtins.
  typedef int arr[strcmp("bar", "foo") + 4 * strncmp("foo", "bar", 4)]; // expected-warning {{variable length array folded to constant array as an extension}} \
                                                                           expected-warning {{variable length arrays in C++ are a Clang extension}} \
                                                                           expected-note {{non-constexpr function 'strcmp' cannot be used in a constant expression}}
  typedef int arr[3];
}

namespace ExplicitInstantiation {
  template<typename T> void f() noexcept {}
  template<typename T> struct X { void f() noexcept {} };
  template void f<int>();
  template void X<int>::f();
}

namespace ConversionFunction {
  struct A { template<typename T> operator T() noexcept; };
  int a = A().operator int();
}

using size_t = decltype(sizeof(0));

namespace OperatorDelete {
  struct W {};
  struct X {};
  struct Y {};
  struct Z {};
  template<bool N, bool D> struct T {};
}
void *operator new(size_t, OperatorDelete::W) noexcept(false);
void operator delete(void*, OperatorDelete::W) noexcept(false) = delete; // expected-note {{here}}
void *operator new(size_t, OperatorDelete::X) noexcept(false);
void operator delete(void*, OperatorDelete::X) noexcept(true) = delete; // expected-note {{here}}
void *operator new(size_t, OperatorDelete::Y) noexcept(true);
void operator delete(void*, OperatorDelete::Y) noexcept(false) = delete; // expected-note {{here}}
void *operator new(size_t, OperatorDelete::Z) noexcept(true);
void operator delete(void*, OperatorDelete::Z) noexcept(true) = delete; // expected-note {{here}}
template<bool N, bool D> void *operator new(size_t, OperatorDelete::T<N, D>) noexcept(N);
template<bool N, bool D> void operator delete(void*, OperatorDelete::T<N, D>) noexcept(D) = delete; // expected-note 4{{here}}
namespace OperatorDelete {
  struct A { A(); };
  A *w = new (W{}) A; // expected-error {{deleted function}}
  A *x = new (X{}) A; // expected-error {{deleted function}}
  A *y = new (Y{}) A; // expected-error {{deleted function}}
  A *z = new (Z{}) A; // expected-error {{deleted function}}

  A *t00 = new (T<false, false>{}) A; // expected-error {{deleted function}}
  A *t01 = new (T<false, true>{}) A; // expected-error {{deleted function}}
  A *t10 = new (T<true, false>{}) A; // expected-error {{deleted function}}
  A *t11 = new (T<true, true>{}) A; // expected-error {{deleted function}}
}
