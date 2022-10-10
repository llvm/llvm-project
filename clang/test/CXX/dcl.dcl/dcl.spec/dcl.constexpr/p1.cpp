// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -verify -std=c++14 %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -verify -std=c++2a %s

// MSVC always adopted the C++17 rule that implies that constexpr variables are
// implicitly inline, so do the test again.
// RUN: %clang_cc1 -triple x86_64-windows-msvc -DMS_ABI -fsyntax-only -verify -std=c++11 %s

struct notlit { // expected-note {{not literal because}}
  notlit() {}
};
struct notlit2 {
  notlit2() {}
};

// valid declarations
constexpr int i1 = 0;
constexpr int f1() { return 0; }
struct s1 {
  constexpr static int mi1 = 0;
  const static int mi2;
};
constexpr int s1::mi2 = 0;

// invalid declarations
// not a definition of an object
constexpr extern int i2; // expected-error {{constexpr variable declaration must be a definition}}
// not a literal type
constexpr notlit nl1; // expected-error {{constexpr variable cannot have non-literal type 'const notlit'}}
// function parameters
void f2(constexpr int i) {} // expected-error {{function parameter cannot be constexpr}}
// non-static member
struct s2 {
  constexpr int mi1; // expected-error {{non-static data member cannot be constexpr; did you intend to make it const?}}
  static constexpr int mi2;
#if __cplusplus <= 201402L && !defined(MS_ABI)
  // expected-error@-2 {{requires an initializer}}
#else
  // expected-error@-4 {{constexpr variable 'mi2' must be initialized by a constant expression}}
#endif
  mutable constexpr int mi3 = 3; // expected-error-re {{non-static data member cannot be constexpr{{$}}}} expected-error {{'mutable' and 'const' cannot be mixed}}
};
// typedef
typedef constexpr int CI; // expected-error {{typedef cannot be constexpr}}
// tag
constexpr class C1 {}; // expected-error {{class cannot be marked constexpr}}
constexpr struct S1 {}; // expected-error {{struct cannot be marked constexpr}}
constexpr union U1 {}; // expected-error {{union cannot be marked constexpr}}
constexpr enum E1 {}; // expected-error {{enum cannot be marked constexpr}}
template <typename T> constexpr class TC1 {}; // expected-error {{class cannot be marked constexpr}}
template <typename T> constexpr struct TS1 {}; // expected-error {{struct cannot be marked constexpr}}
template <typename T> constexpr union TU1 {}; // expected-error {{union cannot be marked constexpr}}
class C2 {} constexpr; // expected-error {{class cannot be marked constexpr}}
struct S2 {} constexpr; // expected-error {{struct cannot be marked constexpr}}
union U2 {} constexpr; // expected-error {{union cannot be marked constexpr}}
enum E2 {} constexpr; // expected-error {{enum cannot be marked constexpr}}
constexpr class C3 {} c3 = C3();
constexpr struct S3 {} s3 = S3();
constexpr union U3 {} u3 = {};
constexpr enum E3 { V3 } e3 = V3;
class C4 {} constexpr c4 = C4();
struct S4 {} constexpr s4 = S4();
union U4 {} constexpr u4 = {};
enum E4 { V4 } constexpr e4 = V4;
constexpr int; // expected-error {{constexpr can only be used in variable and function declarations}}
// redeclaration mismatch
constexpr int f3(); // expected-note {{previous declaration is here}}
int f3(); // expected-error {{non-constexpr declaration of 'f3' follows constexpr declaration}}
int f4(); // expected-note {{previous declaration is here}}
constexpr int f4(); // expected-error {{constexpr declaration of 'f4' follows non-constexpr declaration}}
template<typename T> constexpr T f5(T);
template<typename T> constexpr T f5(T); // expected-note {{previous}}
template<typename T> T f5(T); // expected-error {{non-constexpr declaration of 'f5' follows constexpr declaration}}
template<typename T> T f6(T); // expected-note {{here}}
template<typename T> constexpr T f6(T); // expected-error {{constexpr declaration of 'f6' follows non-constexpr declaration}}
// destructor
struct ConstexprDtor {
  constexpr ~ConstexprDtor() = default;
#if __cplusplus <= 201703L
  // expected-error@-2 {{destructor cannot be declared constexpr}}
#endif
};

// template stuff
template <typename T> constexpr T ft(T t) { return t; }
template <typename T> T gt(T t) { return t; }
struct S {
  template<typename T> constexpr T f(); // expected-warning 0-1{{C++14}} expected-note 0-1{{candidate}}
  template <typename T>
  T g() const; // expected-note-re {{candidate template ignored: could not match 'T (){{( __attribute__\(\(thiscall\)\))?}} const' against 'char (){{( __attribute__\(\(thiscall\)\))?}}'}}
};

// explicit specialization can differ in constepxr
template <> notlit ft(notlit nl) { return nl; }
template <> char ft(char c) { return c; } // expected-note {{previous}}
template <> constexpr char ft(char nl); // expected-error {{constexpr declaration of 'ft<char>' follows non-constexpr declaration}}
template <> constexpr int gt(int nl) { return nl; }
template <> notlit S::f() const { return notlit(); }
#if __cplusplus >= 201402L
// expected-error@-2 {{no function template matches}}
#endif
template <> constexpr int S::g() { return 0; } // expected-note {{previous}}
#if __cplusplus < 201402L
// expected-warning@-2 {{C++14}}
#else
// expected-error@-4 {{does not match any declaration in 'S'}}
#endif
template <> int S::g() const; // expected-error {{non-constexpr declaration of 'g<int>' follows constexpr declaration}}
// specializations can drop the 'constexpr' but not the implied 'const'.
template <> char S::g() { return 0; } // expected-error {{no function template matches}}
template <> double S::g() const { return 0; } // ok

constexpr int i3 = ft(1);

void test() {
  // ignore constexpr when instantiating with non-literal
  notlit2 nl2;
  (void)ft(nl2);
}

// Examples from the standard:
constexpr int square(int x); // expected-note {{declared here}}
constexpr int bufsz = 1024;

constexpr struct pixel { // expected-error {{struct cannot be marked constexpr}}
  int x;
  int y;
  constexpr pixel(int);
};

constexpr pixel::pixel(int a)
  : x(square(a)), y(square(a)) // expected-note {{undefined function 'square' cannot be used in a constant expression}}
  { }

constexpr pixel small(2); // expected-error {{must be initialized by a constant expression}} expected-note {{in call to 'pixel(2)'}}

constexpr int square(int x) {
  return x * x;
}

constexpr pixel large(4);

int next(constexpr int x) { // expected-error {{function parameter cannot be constexpr}}
      return x + 1;
}

extern constexpr int memsz; // expected-error {{constexpr variable declaration must be a definition}}

namespace {
  struct A {
    static constexpr int n = 0;
  };
  // FIXME: We should diagnose this prior to C++17.
  const int &r = A::n;
}
