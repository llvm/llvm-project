// RUN: %clang_cc1 -std=c++1z -verify -triple i686-linux-gnu %s

template<typename T, typename U> struct same;
template<typename T> struct same<T, T> { ~same(); };

struct Empty {};

struct A {
  int a;
};

namespace NonPublicMembers {
  struct NonPublic1 {
  protected:
    int a; // expected-note {{declared protected here}}
  };

  struct NonPublic2 {
  private:
    int a; // expected-note 2{{declared private here}}
  };

  struct NonPublic3 : private A {}; // expected-note {{declared private here}}

  struct NonPublic4 : NonPublic2 {};

  void test() {
    auto [a1] = NonPublic1(); // expected-error {{cannot decompose protected member 'a' of 'NonPublicMembers::NonPublic1'}}
    auto [a2] = NonPublic2(); // expected-error {{cannot decompose private member 'a' of 'NonPublicMembers::NonPublic2'}}
    auto [a3] = NonPublic3(); // expected-error {{cannot decompose members of inaccessible base class 'A' of 'NonPublicMembers::NonPublic3'}}
    auto [a4] = NonPublic4(); // expected-error {{cannot decompose private member 'a' of 'NonPublicMembers::NonPublic2'}}
  }
}

namespace AnonymousMember {
  struct Struct {
    struct { // expected-note {{declared here}}
      int i;
    };
  };

  struct Union {
    union { // expected-note {{declared here}}
      int i;
    };
  };

  void test() {
    auto [a1] = Struct(); // expected-error {{cannot decompose class type 'AnonymousMember::Struct' because it has an anonymous struct member}}
    auto [a2] = Union(); // expected-error {{cannot decompose class type 'AnonymousMember::Union' because it has an anonymous union member}}
  }
}

namespace MultipleClasses {
  struct B : A {
    int a;
  };

  struct C { int a; };
  struct D : A, C {};

  struct E : virtual A {};
  struct F : A, E {}; // expected-warning {{direct base 'A' is inaccessible due to ambiguity}}

  struct G : virtual A {};
  struct H : E, G {};

  struct I { int i; };
  struct J : I {};
  struct K : I, virtual J {}; // expected-warning {{direct base 'MultipleClasses::I' is inaccessible due to ambiguity}}

  struct L : virtual J {};
  struct M : virtual J, L {};

  void test() {
    auto [b] = B(); // expected-error {{cannot decompose class type 'B': both it and its base class 'A' have non-static data members}}
    auto [d] = D(); // expected-error {{cannot decompose class type 'D': its base classes 'A' and 'MultipleClasses::C' have non-static data members}}
    auto [e] = E();
    auto [f] = F(); // expected-error-re {{cannot decompose members of ambiguous base class 'A' of 'F':{{.*}}struct MultipleClasses::F -> struct A{{.*}}struct MultipleClasses::F -> struct MultipleClasses::E -> struct A}}
    auto [h] = H(); // ok, only one (virtual) base subobject even though there are two paths to it
    auto [k] = K(); // expected-error {{cannot decompose members of ambiguous base class 'MultipleClasses::I'}}
    auto [m] = M(); // ok, all paths to I are through the same virtual base subobject J

    same<decltype(m), int>();
  }
}

namespace BindingTypes {
  struct A {
    int i = 0;
    int &r = i;
    const float f = i;
    mutable volatile int mvi;
  };
  void e() {
    auto [i,r,f,mvi] = A();

    same<decltype(i), int>();
    same<decltype(r), int&>();
    same<decltype(f), const float>();
    same<decltype(mvi), volatile int>();

    same<decltype((i)), int&>();
    same<decltype((r)), int&>();
    same<decltype((f)), const float&>();
    same<decltype((mvi)), volatile int&>();
  }
  void f() {
    auto &&[i,r,f,mvi] = A();

    same<decltype(i), int>();
    same<decltype(r), int&>();
    same<decltype(f), const float>();
    same<decltype(mvi), volatile int>();

    same<decltype((i)), int&>();
    same<decltype((r)), int&>();
    same<decltype((f)), const float&>();
    same<decltype((mvi)), volatile int&>();
  }
  void g() {
    const auto [i,r,f,mvi] = A();

    same<decltype(i), const int>();
    same<decltype(r), int&>();
    same<decltype(f), const float>();
    same<decltype(mvi), volatile int>(); // not 'const volatile int', per expected resolution of DRxxx

    same<decltype((i)), const int&>();
    same<decltype((r)), int&>();
    same<decltype((f)), const float&>();
    same<decltype((mvi)), volatile int&>(); // not 'const volatile int&', per expected resolution of DRxxx
  }
  void h() {
    typedef const A CA;
    auto &[i,r,f,mvi] = CA(); // type of var is 'const A &'

    same<decltype(i), const int>(); // not 'int', per expected resolution of DRxxx
    same<decltype(r), int&>();
    same<decltype(f), const float>();
    same<decltype(mvi), volatile int>(); // not 'const volatile int', per expected resolution of DRxxx

    same<decltype((i)), const int&>(); // not 'int&', per expected resolution of DRxxx
    same<decltype((r)), int&>();
    same<decltype((f)), const float&>();
    same<decltype((mvi)), volatile int&>(); // not 'const volatile int&', per expected resolution of DRxxx
  }
  struct B {
    mutable int i;
  };
  void mut() {
    auto [i] = B();
    const auto [ci] = B();
    volatile auto [vi] = B();
    same<decltype(i), int>();
    same<decltype(ci), int>();
    same<decltype(vi), volatile int>();
  }
}

namespace Bitfield {
  struct S { unsigned long long x : 4, y : 32; int z; }; // expected-note 2{{here}}
  int f(S s) {
    auto [a, b, c] = s;
    unsigned long long &ra = a; // expected-error {{bit-field 'x'}}
    unsigned long long &rb = b; // expected-error {{bit-field 'y'}}
    int &rc = c;

    // the type of the binding is the type of the field
    same<decltype(a), unsigned long long>();
    same<decltype(b), unsigned long long>();

    // the type of the expression is an lvalue of the field type
    // (even though a reference can't bind to the field)
    same<decltype((a)), unsigned long long&>();
    same<decltype((b)), unsigned long long&>();

    // the expression promotes to a type large enough to hold the result
    same<decltype(+a), int>();
    same<decltype(+b), unsigned int>();
    return rc;
  }
}

namespace Constexpr {
  struct Q { int a, b; constexpr Q() : a(1), b(2) {} };
  constexpr Q q;
  auto &[qa, qb] = q;
  static_assert(&qa == &q.a && &qb == &q.b);
  static_assert(qa == 1 && qb == 2);
}

namespace std_example {
  struct S { int x1 : 2; volatile double y1; };
  S f();
  const auto [x, y] = f();

  same<decltype((x)), const int&> same1;
  same<decltype((y)), const volatile double&> same2;
}

namespace p0969r0 {
  struct A {
    int x;
    int y;
  };
  struct B : private A { // expected-note {{declared private here}}
    void test_member() {
      auto &[x, y] = *this;
    }
    friend void test_friend(B);
  };
  void test_friend(B b) {
    auto &[x, y] = b;
  }
  void test_external(B b) {
    auto &[x, y] = b; // expected-error {{cannot decompose members of inaccessible base class 'p0969r0::A' of 'p0969r0::B'}}
  }

  struct C {
    int x;
  protected:
    int y; // expected-note {{declared protected here}} expected-note {{can only access this member on an object of type 'p0969r0::D'}}
    void test_member() {
      auto &[x, y] = *this;
    }
    friend void test_friend(struct D);
  };
  struct D : C {
    static void test_member(D d, C c) {
      auto &[x1, y1] = d;
      auto &[x2, y2] = c; // expected-error {{cannot decompose protected member 'y' of 'p0969r0::C'}}
    }
  };
  void test_friend(D d) {
    auto &[x, y] = d;
  }
  void test_external(D d) {
    auto &[x, y] = d; // expected-error {{cannot decompose protected member 'y' of 'p0969r0::C'}}
  }
}
