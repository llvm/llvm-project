// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s

// p1099 'using SCOPEDENUM::MEMBER;'

namespace Zero {
namespace Bob {
enum class Kevin {
  Stuart,
  AlsoStuart
#if __cplusplus >= 202002L
// expected-note@-3{{target of using declaration}}
// expected-note@-3{{target of using declaration}}
#endif
};
} // namespace Bob

using Bob::Kevin::Stuart;
#if __cplusplus < 202002L
// expected-warning@-2{{using declaration naming a scoped enumerator is a C++20 extension}}
#else
using Bob::Kevin::Stuart;

auto b = Stuart;

namespace Foo {
int Stuart;               // expected-note{{conflicting declaration}}
using Bob::Kevin::Stuart; // expected-error{{target of using declaration conflicts}}

using Bob::Kevin::AlsoStuart; // expected-note{{using declaration}}
int AlsoStuart;               // expected-error{{declaration conflicts with target}}
} // namespace Foo
#endif

} // namespace Zero

namespace One {

// derived from [namespace.udecl]/3
enum class button { up,
                    down };
struct S {
  using button::up;
#if __cplusplus < 202002L
  // expected-warning@-2{{a C++20 extension}}
  // expected-error@-3{{using declaration in class}}
#else
  button b = up;
#endif
};

#if __cplusplus >= 202002L
// some more
struct T : S {
  button c = up;
};
#endif
enum E2 { e2 };
} // namespace One

namespace Two {
enum class E1 { e1 };

struct S {
  using One::e2;
#if __cplusplus < 202002L
  // expected-error@-2{{using declaration in class}}
#else
  One::E2 c = e2;
#endif
};

} // namespace Two

namespace Three {

enum E3 { e3 };
struct e3;

struct S {
  using Three::e3; // expected-error{{using declaration in class}}

  enum class E4 { e4 };
  enum E5 { e5 };
};

using S::e5;
using S::E4::e4;
#if __cplusplus < 202002L
// expected-error@-3{{using declaration cannot refer to class member}}
// expected-note@-4{{use a constexpr variable instead}}
// expected-warning@-4{{a C++20 extension}}
// expected-error@-5{{using declaration cannot refer to class member}}
// expected-note@-6{{use a constexpr variable instead}}
#else
auto a = e4;
auto b = e5;
#endif
} // namespace Three

namespace Four {

template <typename T>
struct TPL {
  enum class E1 { e1 };
  struct IN {
    enum class E2 { e2 };
  };

protected:
  enum class E3 { e3 }; // expected-note{{declared protected here}}
};

using TPL<int>::E1::e1;
#if __cplusplus < 202002L
// expected-warning@-2{{a C++20 extension}}
// expected-error@-3{{using declaration cannot refer to class member}}
// expected-note@-4{{use a constexpr variable instead}}
#else
using TPL<float>::IN::E2::e2;

auto a = e1;
auto b = e2;
#endif

enum class E4 { e4 };
template <typename T>
struct DER : TPL<int> {
  using TPL<T>::E1::e1;
#if __cplusplus < 202002L
  // expected-warning@-2{{a C++20 extension}}
  // expected-warning@-3{{using declaration naming a scoped}}
  // expected-error@-4{{which is not a base}}
#endif
  using TPL<T>::E3::e3; // expected-error{{is a protected member}}
#if __cplusplus < 202002L
  // expected-warning@-2 2{{using declaration naming a scoped}}
  // expected-error@-3{{which is not a base}}
#endif

  using E4::e4;
#if __cplusplus < 202002L
  // expected-warning@-2{{a C++20 extension}}
  // expected-error@-3{{which is not a class}}
#else
  auto Foo() { return e1; }
  auto Bar() { return e2; }
#endif
};

DER<float> x; // expected-note{{requested here}}
DER<int> y;
#if __cplusplus < 202002L
// expected-note@-2{{requested here}}
#else
auto y1 = y.Foo();
auto y2 = y.Bar();
#endif
} // namespace Four

namespace Five {
template <unsigned I, unsigned K>
struct Quux {
  enum class Q : unsigned; // expected-note{{member is declared here}}
  enum class R : unsigned { i = I,
                            k = K };
};

using Quux<1, 2>::Q::nothing; // expected-error{{implicit instantiation of undefined}}
using Quux<1, 2>::R::i;
#if __cplusplus < 202002L
// expected-warning@-2{{a C++20 extension}}
// expected-error@-3{{using declaration cannot refer to class member}}
// expected-note@-4{{use a constexpr variable instead}}
#endif

} // namespace Five

namespace Six {
template <unsigned I, unsigned K>
struct Quux {
  enum class Q : unsigned; // expected-note{{member is declared here}}
  enum class R : unsigned { i = I,
                            k = K };
};

template <unsigned I> struct Fido {
  using Quux<I, I>::Q::nothing; // expected-error{{implicit instantiation of undefined}}
};

Fido<2> a; // expected-note{{in instantiation}}

} // namespace Six

namespace Seven {
template <unsigned I, unsigned K>
struct Quux {
  enum class R : unsigned { i = I,
                            k = K };
};

template <unsigned I> struct Toto {
  using Quux<I, I>::R::i;
#if __cplusplus < 202002L
  // expected-warning@-2{{a C++20 extension}}
// expected-error@-3{{refers into}}
#else
  static_assert(unsigned(i) == I);
#endif
};

Toto<2> b;
#if __cplusplus < 202002L
// expected-note@-2{{in instantiation}}
#endif

} // namespace Seven

namespace Eight {
struct Kevin {
  enum class B { a };
  enum a {};
};

using Kevin::B::a;
#if __cplusplus < 202002L
// expected-warning@-2{{a C++20 extension}}
// expected-error@-3{{using declaration cannot refer to class member}}
// expected-note@-4{{use a constexpr variable instead}}
#endif
using Kevin::B::a;
#if __cplusplus < 202002L
// expected-warning@-2{{a C++20 extension}}
// expected-error@-3{{using declaration cannot refer to class member}}
// expected-note@-4{{use a constexpr variable instead}}
#endif

class X : Kevin {
  using Kevin::B::a; // expected-note{{previous using declaration}}
#if __cplusplus < 202002L
// expected-warning@-2{{a C++20 extension}}
#endif
  using Kevin::a;
  using Kevin::B::a; // expected-error{{redeclaration of using declaration}}
};

} // namespace Eight

namespace Nine {
namespace Q {
enum class Bob { a };
using Bob::a;
#if __cplusplus < 202002L
// expected-warning@-2{{a C++20 extension}}
#endif
} // namespace Q

using Q::a;
using Q::Bob::a;
#if __cplusplus < 202002L
// expected-warning@-2{{a C++20 extension}}
#endif

#if __cplusplus >= 202002L
struct Foo {
  using Q::a; // expected-note{{previous using declaration}}
  using Q::Bob::a;
  using Q::a; // expected-error{{redeclaration of using declaration}}
};
#endif
} // namespace Nine
