// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple=x86_64-pc-win32 -fms-compatibility %s
// expected-no-diagnostics

// GCC and MSVC allows '__restrict' in the cv-qualifier-seq of member functions
// but ignore it for the purpose of matching and type comparisons. To match this
// behaviour, we always strip restrict from pointers to member functions as well
// as from member function declarations.

template <typename A, typename B>
struct is_same {
    static constexpr bool value = false;
};

template <typename A>
struct is_same<A, A> {
    static constexpr bool value = true;
};

struct S {
  void a() __restrict;
  void b() __restrict;
  void c();
};

void S::a() __restrict { }
void S::b() { }
void S::c() __restrict { }

void (S::*p1)() __restrict = &S::a;
void (S::*p2)()            = &S::a;
void (S::*p3)() __restrict = &S::c;
void (S::*p4)()            = &S::c;

static_assert(__is_same(void (S::*) (), void (S::*) () __restrict));
static_assert(__is_same(void (S::*) () const &, void (S::*) () __restrict const &));
static_assert(__is_same(void (S::*) () volatile &&, void (S::*) () volatile __restrict &&));

static_assert(__is_same(decltype(&S::a), void (S::*) () __restrict));
static_assert(__is_same(decltype(&S::a), void (S::*) ()));
static_assert(__is_same(decltype(&S::b), void (S::*) () __restrict));
static_assert(__is_same(decltype(&S::b), void (S::*) ()));
static_assert(__is_same(decltype(&S::c), void (S::*) () __restrict));
static_assert(__is_same(decltype(&S::c), void (S::*) ()));

template <typename>
struct TS {
    void a() __restrict;
    void b() __restrict;
    void c();
};

template <typename T>
void TS<T>::b() __restrict { }

template <typename T>
void TS<T>::c() { }

void (TS<int>::*p5)() __restrict = &TS<int>::a;
void (TS<int>::*p6)()            = &TS<int>::a;
void (TS<int>::*p7)() __restrict = &TS<int>::c;
void (TS<int>::*p8)()            = &TS<int>::c;

void h() {
    TS<int>().a();
    TS<int>().b();
    TS<int>().c();
    TS<double>().a();
    TS<double>().b();
    TS<double>().c();
}

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
