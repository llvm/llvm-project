// RUN: %clang_cc1 -verify -std=c++20 %s

template <int N>
concept C0 = (N == 0);
template <int N>
concept C1 = (N == 1);
template <int N>
concept C2 = (N == 2);

// Checks are indexed by:
// Definition:
//  1. Explicitly defaulted definition
//  2. Deleted definition
//  3. User provided definition
// We have a less constrained user provided method that should not disable
// the (copyable) triviality of the type.

// Note that because Clang does not implement DRs 1496 and 1734, we say some
// classes are trivial when the SMFs are deleted.

template <int N>
struct DefaultConstructorChecker {
    DefaultConstructorChecker() requires C0<N> = default;
    DefaultConstructorChecker() requires C1<N> = delete;
    DefaultConstructorChecker() requires C2<N>;
    DefaultConstructorChecker();
};
static_assert(__is_trivially_copyable(DefaultConstructorChecker<0>));
static_assert(__is_trivially_copyable(DefaultConstructorChecker<1>));
static_assert(__is_trivially_copyable(DefaultConstructorChecker<2>));
static_assert(__is_trivially_copyable(DefaultConstructorChecker<3>));
static_assert(__is_trivial(DefaultConstructorChecker<0>));
// FIXME: DR1496
static_assert(__is_trivial(DefaultConstructorChecker<1>));
static_assert(!__is_trivial(DefaultConstructorChecker<2>));
static_assert(!__is_trivial(DefaultConstructorChecker<3>));

template <int N>
struct CopyConstructorChecker {
    CopyConstructorChecker(const CopyConstructorChecker&) requires C0<N> = default;
    CopyConstructorChecker(const CopyConstructorChecker&) requires C1<N> = delete;
    CopyConstructorChecker(const CopyConstructorChecker&) requires C2<N>;
    CopyConstructorChecker(const CopyConstructorChecker&);
};

static_assert(__is_trivially_copyable(CopyConstructorChecker<0>));
// FIXME: DR1734
static_assert(__is_trivially_copyable(CopyConstructorChecker<1>));
static_assert(!__is_trivially_copyable(CopyConstructorChecker<2>));
static_assert(!__is_trivially_copyable(CopyConstructorChecker<3>));
static_assert(!__is_trivial(CopyConstructorChecker<0>));
static_assert(!__is_trivial(CopyConstructorChecker<1>));
static_assert(!__is_trivial(CopyConstructorChecker<2>));
static_assert(!__is_trivial(CopyConstructorChecker<3>));

template <int N>
struct MoveConstructorChecker {
    MoveConstructorChecker(MoveConstructorChecker&&) requires C0<N> = default;
    MoveConstructorChecker(MoveConstructorChecker&&) requires C1<N> = delete;
    MoveConstructorChecker(MoveConstructorChecker&&) requires C2<N>;
    MoveConstructorChecker(MoveConstructorChecker&&);
};

static_assert(__is_trivially_copyable(MoveConstructorChecker<0>));
// FIXME: DR1734
static_assert(__is_trivially_copyable(MoveConstructorChecker<1>));
static_assert(!__is_trivially_copyable(MoveConstructorChecker<2>));
static_assert(!__is_trivially_copyable(MoveConstructorChecker<3>));
static_assert(!__is_trivial(MoveConstructorChecker<0>));
static_assert(!__is_trivial(MoveConstructorChecker<1>));
static_assert(!__is_trivial(MoveConstructorChecker<2>));
static_assert(!__is_trivial(MoveConstructorChecker<3>));

template <int N>
struct MoveAssignmentChecker {
    MoveAssignmentChecker& operator=(MoveAssignmentChecker&&) requires C0<N> = default;
    MoveAssignmentChecker& operator=(MoveAssignmentChecker&&) requires C1<N> = delete;
    MoveAssignmentChecker& operator=(MoveAssignmentChecker&&) requires C2<N>;
    MoveAssignmentChecker& operator=(MoveAssignmentChecker&&);
};

static_assert(__is_trivially_copyable(MoveAssignmentChecker<0>));
// FIXME: DR1734.
static_assert(__is_trivially_copyable(MoveAssignmentChecker<1>));
static_assert(!__is_trivially_copyable(MoveAssignmentChecker<2>));
static_assert(!__is_trivially_copyable(MoveAssignmentChecker<3>));
static_assert(__is_trivial(MoveAssignmentChecker<0>));
// FIXME: DR1734.
static_assert(__is_trivial(MoveAssignmentChecker<1>));
static_assert(!__is_trivial(MoveAssignmentChecker<2>));
static_assert(!__is_trivial(MoveAssignmentChecker<3>));

template <int N>
struct CopyAssignmentChecker {
    CopyAssignmentChecker& operator=(const CopyAssignmentChecker&) requires C0<N> = default;
    CopyAssignmentChecker& operator=(const CopyAssignmentChecker&) requires C1<N> = delete;
    CopyAssignmentChecker& operator=(const CopyAssignmentChecker&) requires C2<N>;
    CopyAssignmentChecker& operator=(const CopyAssignmentChecker&);
};

static_assert(__is_trivially_copyable(CopyAssignmentChecker<0>));
// FIXME: DR1734.
static_assert(__is_trivially_copyable(CopyAssignmentChecker<1>));
static_assert(!__is_trivially_copyable(CopyAssignmentChecker<2>));
static_assert(!__is_trivially_copyable(CopyAssignmentChecker<3>));
static_assert(__is_trivial(CopyAssignmentChecker<0>));
// FIXME: DR1734.
static_assert(__is_trivial(CopyAssignmentChecker<1>));
static_assert(!__is_trivial(CopyAssignmentChecker<2>));
static_assert(!__is_trivial(CopyAssignmentChecker<3>));


template <int N>
struct KindComparisonChecker1 {
    KindComparisonChecker1& operator=(const KindComparisonChecker1&) requires C0<N> = default;
    KindComparisonChecker1& operator=(KindComparisonChecker1&);
};

template <int N>
struct KindComparisonChecker2 {
    KindComparisonChecker2& operator=(const KindComparisonChecker2&) requires C0<N> = default;
    const KindComparisonChecker2& operator=(KindComparisonChecker2&) const;
};

template <int N>
struct KindComparisonChecker3 {
    using Alias = KindComparisonChecker3;
    Alias& operator=(const Alias&) requires C0<N> = default;
    KindComparisonChecker3& operator=(const KindComparisonChecker3&);
};

static_assert(!__is_trivial(KindComparisonChecker1<0>));
static_assert(!__is_trivially_copyable(KindComparisonChecker1<0>));

static_assert(!__is_trivial(KindComparisonChecker2<0>));
static_assert(!__is_trivially_copyable(KindComparisonChecker2<0>));

static_assert(__is_trivial(KindComparisonChecker3<0>));
static_assert(__is_trivially_copyable(KindComparisonChecker3<0>));

template <class T>
concept HasA = requires(T t) {
    { t.a() };
};

template <class T>
concept HasAB = HasA<T> && requires(T t) {
    { t.b() };
};

template <class T>
concept HasAC = HasA<T> && requires(T t) {
    { t.c() };
};

template <class T>
concept HasABC = HasAB<T> && HasAC<T> && requires(T t) {
    { t.c() };
};

template <class T>
struct ComplexConstraints {
    ComplexConstraints() requires HasABC<T> = default;
    ComplexConstraints() requires HasAB<T>;
    ComplexConstraints() requires HasAC<T>;
    ComplexConstraints() requires HasA<T> = delete;
    ComplexConstraints();
};

struct A {
    void a();
};

struct AB {
    void a();
    void b();
};

struct ABC {
    void a();
    void b();
    void c();
};

struct AC {
    void a();
    void c();
};

static_assert(__is_trivial(ComplexConstraints<ABC>), "");
static_assert(!__is_trivial(ComplexConstraints<AB>), "");
static_assert(!__is_trivial(ComplexConstraints<AC>), "");
static_assert(__is_trivial(ComplexConstraints<A>), "");
static_assert(!__is_trivial(ComplexConstraints<int>), "");


// This is evaluated at the completion of CRTPBase, while `T` is not yet completed.
// This is probably correct behavior.
template <class T>
struct CRTPBase {
  CRTPBase() requires (sizeof(T) > 0);
  CRTPBase() = default;
};

struct Child : CRTPBase<Child> { int x; };
static Child c;


namespace GH57046 {
template<unsigned N>
struct Foo {
  Foo() requires (N==1) {} // expected-note {{declared here}}
  Foo() requires (N==2) = default;
};

template <unsigned N, unsigned M>
struct S {
  Foo<M> data;
  S() requires (N==1) {}
  consteval S() requires (N==2) = default; // expected-note {{non-constexpr constructor 'Foo' cannot be used in a constant expression}}
};

void func() {
  S<2, 1> s1; // expected-error {{is not a constant expression}} expected-note {{in call to 'S()'}}
  S<2, 2> s2;
}
}

namespace GH59206 {

struct A {
  A() = default; //eligible, second constructor unsatisfied
  template<class... Args>
  A(Args&&... args) requires (sizeof...(Args) > 0) {}
};

struct B {
  B() = default; //ineligible, second constructor more constrained
  template<class... Args>
  B(Args&&... args) requires (sizeof...(Args) == 0) {}
};

struct C {
  C() = default; //eligible, but
  template<class... Args> //also eligible and non-trivial
  C(Args&&... args) {}
};

struct D : B {};

static_assert(__is_trivially_copyable(A), "");
static_assert(__is_trivially_copyable(B), "");
static_assert(__is_trivially_copyable(C), "");
static_assert(__is_trivially_copyable(D), "");

// FIXME: Update when https://github.com/llvm/llvm-project/issues/59206 is
// resolved.
static_assert(!__is_trivial(A), "");
static_assert(!__is_trivial(B), "");
static_assert(!__is_trivial(C), "");
static_assert(__is_trivial(D), "");
static_assert(__is_trivially_constructible(A), "");
static_assert(__is_trivially_constructible(B), "");
static_assert(__is_trivially_constructible(C), "");
static_assert(__is_trivially_constructible(D), "");

}

namespace GH60697 {

template <class T>
struct X {
    X() requires false = default;
};
static_assert(!__is_trivial(X<int>));

template <class T>
struct S {
    S() requires(__is_trivially_constructible(T)) = default;

    S() requires(!__is_trivially_constructible(T) &&
                  __is_constructible(T)) {}

    T t;
};

struct D {
    D(int i) : i(i) {}
    int i;
};
static_assert(!__is_trivially_constructible(D));
static_assert(!__is_constructible(D));
static_assert(!__is_trivial(D));

static_assert(!__is_trivially_constructible(S<D>));
static_assert(!__is_constructible(S<D>));

static_assert(__is_trivial(S<int>));
static_assert(!__is_trivial(S<D>));

}

namespace GH62555 {

template <bool B>
struct ExplicitTemplateArgs {
    ExplicitTemplateArgs(ExplicitTemplateArgs&&) = default;
    ExplicitTemplateArgs(ExplicitTemplateArgs<false>&&) requires B {};
};

static_assert(__is_trivially_copyable(ExplicitTemplateArgs<false>));
static_assert(__is_trivially_copyable(ExplicitTemplateArgs<true>));

}
