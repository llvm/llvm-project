// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple=x86_64-pc-win32 -fms-compatibility %s
// expected-no-diagnostics

struct S{ void a(); };

template <typename A, typename B>
struct is_same {
    static constexpr bool value = false;
};

template <typename A>
struct is_same<A, A> {
    static constexpr bool value = true;
};

// GCC and MSVC allows '__restrict' in the cv-qualifier-seq of member functions
// but ignore it for the purpose of matching and type comparisons.
static_assert(__is_same(void (S::*) (), void (S::*) () __restrict));
static_assert(__is_same(void (S::*) () &, void (S::*) () __restrict &));
static_assert(__is_same(void (S::*) () &&, void (S::*) () __restrict &&));
static_assert(is_same<void (S::*) (), void (S::*) () __restrict>::value);
static_assert(is_same<void (S::*) () &, void (S::*) () __restrict &>::value);
static_assert(is_same<void (S::*) () &&, void (S::*) () __restrict &&>::value);

// Non-member function types with '__restrict' are distinct types.
using A = void () __restrict;
using B = void ();
static_assert(!is_same<A, B>::value);

namespace gh11039 {
class foo {
  int member[4];

  void bar(int * a);
};

void foo::bar(int * a) __restrict {
  member[3] = *a;
}
}