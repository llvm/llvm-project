// RUN: %clang_cc1 -verify -fsyntax-only -std=c++11 -pedantic-errors -triple x86_64-linux-gnu %s

// Make sure we know these are legitimate commas and not typos for ';'.
namespace Commas {
  int a,
  b [[ ]],
  c alignas(double);
}

struct S {};
enum E { e, };

auto f() -> struct S {
  return S();
}
auto g() -> enum E {
  return E();
}

int decltype(f())::*ptr_mem_decltype;

class ExtraSemiAfterMemFn {
  // Due to a peculiarity in the C++11 grammar, a deleted or defaulted function
  // is permitted to be followed by either one or two semicolons.
  void f() = delete // expected-error {{expected ';' after delete}}
  void g() = delete; // ok
  void h() = delete;; // ok
  void i() = delete;;; // expected-error {{extra ';' after member function definition}}
};

int *const const p = 0; // expected-error {{duplicate 'const' declaration specifier}}
const const int *q = 0; // expected-error {{duplicate 'const' declaration specifier}}

struct MultiCV {
  void f() const const; // expected-error {{duplicate 'const' declaration specifier}}
};

static_assert(something, ""); // expected-error {{undeclared identifier}}

// PR9903
struct SS {
  typedef void d() = default; // expected-error {{function definition declared 'typedef'}} expected-error {{only special member functions may be defaulted}}
};

using PR14855 = int S::; // expected-error {{expected ';' after alias declaration}}

// Ensure that 'this' has a const-qualified type in a trailing return type for
// a constexpr function.
struct ConstexprTrailingReturn {
  int n;
  constexpr auto f() const -> decltype((n));
};
constexpr const int &ConstexprTrailingReturn::f() const { return n; }

namespace TestIsValidAfterTypeSpecifier {
struct s {} v;

struct s
thread_local tl;

struct s
&r0 = v;

struct s
&&r1 = s();

struct s
bitand r2 = v;

struct s
and r3 = s();

enum E {};
enum E
[[]] e;

}

namespace PR5066 {
  using T = int (*f)(); // expected-error {{type-id cannot have a name}}
  template<typename T> using U = int (*f)(); // expected-error {{type-id cannot have a name}}
  auto f() -> int (*f)(); // expected-error {{only variables can be initialized}} expected-error {{expected ';'}}
  auto g = []() -> int (*f)() {}; // expected-error {{type-id cannot have a name}}
}

namespace FinalOverride {
  struct Base {
    virtual void *f();
    virtual void *g();
    virtual void *h();
    virtual void *i();
  };
  struct Derived : Base {
    virtual auto f() -> void *final;
    virtual auto g() -> void *override;
    virtual auto h() -> void *final override;
    virtual auto i() -> void *override final;
  };
}

namespace UsingDeclAttrs {
  using T __attribute__((aligned(1))) = int;
  using T [[gnu::aligned(1)]] = int;
  static_assert(alignof(T) == 1, "");

  using [[gnu::aligned(1)]] T = int; // expected-error {{an attribute list cannot appear here}}
  using T = int [[gnu::aligned(1)]]; // expected-error {{'aligned' attribute cannot be applied to types}}
}

namespace DuplicateSpecifier {
  constexpr constexpr int f(); // expected-warning {{duplicate 'constexpr' declaration specifier}}
  constexpr int constexpr a = 0; // expected-warning {{duplicate 'constexpr' declaration specifier}}

  struct A {
    friend constexpr int constexpr friend f(); // expected-warning {{duplicate 'friend' declaration specifier}} \
                                               // expected-warning {{duplicate 'constexpr' declaration specifier}}
    friend struct A friend; // expected-warning {{duplicate 'friend'}} expected-error {{'friend' must appear first}}
  };
}

namespace ColonColonDecltype {
  struct S { struct T {}; };
  ::decltype(S())::T invalid; // expected-error {{expected unqualified-id}}
}

namespace AliasDeclEndLocation {
  template<typename T> struct A {};
  // Ensure that we correctly determine the end of this declaration to be the
  // end of the annotation token, not the beginning.
  using B = AliasDeclEndLocation::A<int
    > // expected-error {{expected ';' after alias declaration}}
    +;
  using C = AliasDeclEndLocation::A<int
    >\
> // expected-error {{expected ';' after alias declaration}}
    ;
  using D = AliasDeclEndLocation::A<int
    > // expected-error {{expected ';' after alias declaration}}
  // FIXME: After splitting this >> into two > tokens, we incorrectly determine
  // the end of the template-id to be after the *second* '>'.
  using E = AliasDeclEndLocation::A<int>>;
#define GGG >>>
  using F = AliasDeclEndLocation::A<int GGG;
  // expected-error@-1 {{expected ';' after alias declaration}}
  B something_else;
}

struct Base { virtual void f() = 0; virtual void g() = 0; virtual void h() = 0; };
struct MemberComponentOrder : Base {
  void f() override __asm__("foobar") __attribute__(( )) {}
  void g() __attribute__(( )) override;
  void h() __attribute__(( )) override {}
};

void NoMissingSemicolonHere(struct S
                            [3]);
template<int ...N> void NoMissingSemicolonHereEither(struct S
                                                     ... [N]);

// This must be at the end of the file; we used to look ahead past the EOF token here.
// expected-error@+1 {{expected unqualified-id}} expected-error@+1{{expected ';'}}
using
