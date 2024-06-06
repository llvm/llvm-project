// RUN: %clang_cc1 -std=c++14 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++14 -verify=ref,both %s

namespace MemberPointers {
  struct A {
    constexpr A(int n) : n(n) {}
    int n;
    constexpr int f() const { return n + 3; }
  };

  constexpr A a(7);
  static_assert(A(5).*&A::n == 5, "");
  static_assert((&a)->*&A::n == 7, "");
  static_assert((A(8).*&A::f)() == 11, "");
  static_assert(((&a)->*&A::f)() == 10, "");

  struct B : A {
    constexpr B(int n, int m) : A(n), m(m) {}
    int m;
    constexpr int g() const { return n + m + 1; }
  };
  constexpr B b(9, 13);
  static_assert(B(4, 11).*&A::n == 4, "");
  static_assert(B(4, 11).*&B::m == 11, "");
  static_assert(B(4, 11).m == 11, "");
  static_assert(B(4, 11).*(int(A::*))&B::m == 11, "");
  static_assert(B(4, 11).*&B::m == 11, "");
  static_assert((&b)->*&A::n == 9, "");
  static_assert((&b)->*&B::m == 13, "");
  static_assert((&b)->*(int(A::*))&B::m == 13, "");
  static_assert((B(4, 11).*&A::f)() == 7, "");
  static_assert((B(4, 11).*&B::g)() == 16, "");

  static_assert((B(4, 11).*(int(A::*)() const)&B::g)() == 16, "");

  static_assert(((&b)->*&A::f)() == 12, "");
  static_assert(((&b)->*&B::g)() == 23, "");
  static_assert(((&b)->*(int(A::*)()const)&B::g)() == 23, "");


  struct S {
    constexpr S(int m, int n, int (S::*pf)() const, int S::*pn) :
      m(m), n(n), pf(pf), pn(pn) {}
    constexpr S() : m(), n(), pf(&S::f), pn(&S::n) {}

    constexpr int f() const { return this->*pn; }
    virtual int g() const;

    int m, n;
    int (S::*pf)() const;
    int S::*pn;
  };

  constexpr int S::*pm = &S::m;
  constexpr int S::*pn = &S::n;

  constexpr int (S::*pf)() const = &S::f;
  constexpr int (S::*pg)() const = &S::g;

  constexpr S s(2, 5, &S::f, &S::m);

  static_assert((s.*&S::f)() == 2, "");
  static_assert((s.*s.pf)() == 2, "");

  static_assert(pf == &S::f, "");

  static_assert(pf == s.*&S::pf, "");

  static_assert(pm == &S::m, "");
  static_assert(pm != pn, "");
  static_assert(s.pn != pn, "");
  static_assert(s.pn == pm, "");
  static_assert(pg != nullptr, "");
  static_assert(pf != nullptr, "");
  static_assert((int S::*)nullptr == nullptr, "");
  static_assert(pg == pg, ""); // both-error {{constant expression}} \
                               // both-note {{comparison of pointer to virtual member function 'g' has unspecified value}}
  static_assert(pf != pg, ""); // both-error {{constant expression}} \
                               // both-note {{comparison of pointer to virtual member function 'g' has unspecified value}}

  template<int n> struct T : T<n-1> { const int X = n;};
  template<> struct T<0> { int n; char k;};
  template<> struct T<30> : T<29> { int m; };

  T<17> t17;
  T<30> t30;

  constexpr int (T<15>::*deepm) = (int(T<10>::*))&T<30>::m;
  constexpr int (T<10>::*deepn) = &T<0>::n;
  constexpr char (T<10>::*deepk) = &T<0>::k;

  static_assert(&(t17.*deepn) == &t17.n, "");
  static_assert(&(t17.*deepk) == &t17.k, "");
  static_assert(deepn == &T<2>::n, "");

  constexpr int *pgood = &(t30.*deepm);
  constexpr int *pbad = &(t17.*deepm); // both-error {{constant expression}}
  static_assert(&(t30.*deepm) == &t30.m, "");

  static_assert(deepm == &T<50>::m, "");
  static_assert(deepm != deepn, "");

  constexpr T<5> *p17_5 = &t17;
  constexpr T<13> *p17_13 = (T<13>*)p17_5;
  constexpr T<23> *p17_23 = (T<23>*)p17_13; // both-error {{constant expression}} \
                                            // both-note {{cannot cast object of dynamic type 'T<17>' to type 'T<23>'}}
  constexpr T<18> *p17_18 = (T<18>*)p17_13; // both-error {{constant expression}} \
                                            // both-note {{cannot cast object of dynamic type 'T<17>' to type 'T<18>'}}
  static_assert(&(p17_5->*(int(T<0>::*))deepn) == &t17.n, "");
  static_assert(&(p17_5->*(int(T<0>::*))deepn), "");


  static_assert(&(p17_13->*deepn) == &t17.n, "");
  constexpr int *pbad2 = &(p17_13->*(int(T<9>::*))deepm); // both-error {{constant expression}}

  constexpr T<5> *p30_5 = &t30;
  constexpr T<23> *p30_23 = (T<23>*)p30_5;
  constexpr T<13> *p30_13 = p30_23;
  static_assert(&(p30_13->*deepn) == &t30.n, "");
  static_assert(&(p30_23->*deepn) == &t30.n, "");
  static_assert(&(p30_5->*(int(T<3>::*))deepn) == &t30.n, "");

  static_assert(&(p30_5->*(int(T<2>::*))deepm) == &t30.m, "");
  static_assert(&(((T<17>*)p30_13)->*deepm) == &t30.m, "");
  static_assert(&(p30_23->*deepm) == &t30.m, "");


  /// Added tests not from constant-expression-cxx11.cpp
  static_assert(pm, "");
  static_assert(!((int S::*)nullptr), "");
  constexpr int S::*pk = nullptr;
  static_assert(!pk, "");
}

namespace test3 {
  struct nsCSSRect {
  };
  static int nsCSSRect::* sides;
  nsCSSRect dimenX;
  void ParseBoxCornerRadii(int y) {
    switch (y) {
    }
    int& x = dimenX.*sides;
  }
}

void foo() {
  class X;
  void (X::*d) ();
  d = nullptr; /// This calls in the constant interpreter.
}

namespace {
  struct A { int n; };
  struct B { int n; };
  struct C : A, B {};
  struct D { double d; C c; };
  const int &&u = static_cast<B&&>(0, ((D&&)D{}).*&D::c).n; // both-warning {{left operand of comma operator has no effect}}
}

/// From SemaTemplate/instantiate-member-pointers.cpp
namespace {
  struct Y {
    int x;
  };

  template<typename T, typename Class, T Class::*Ptr>
  struct X3 {
    X3<T, Class, Ptr> &operator=(const T& value) {
      return *this;
    }
  };

  typedef int Y::*IntMember;
  template<IntMember Member>
  struct X4 {
    X3<int, Y, Member> member;
    int &getMember(Y& y) { return y.*Member; }
  };

  int &get_X4(X4<&Y::x> x4, Y& y) {
    return x4.getMember(y);
  }
}

/// From test/CXX/basic/basic.def.odr/p2.cpp
namespace {
  void use(int);
  struct S { int x; int f() const; };
  constexpr S *ps = nullptr;
  S *const &psr = ps;

  void test() {
    use(ps->*&S::x);
    use(psr->*&S::x);
  }
}
