// RUN: %clang_cc1 -std=c++11 -verify %s

// rdar://12240916 stack overflow.
namespace rdar12240916 {

struct S2 {
  S2(const S2&);
  S2();
};

struct S { // expected-note {{not complete}}
  S x; // expected-error {{incomplete type}}
  S2 y;
};

S foo() {
  S s;
  return s;
}

struct S3; // expected-note {{forward declaration}}

struct S4 {
  S3 x; // expected-error {{incomplete type}}
  S2 y;
};

struct S3 {
  S4 x;
  S2 y;
};

S4 foo2() {
  S4 s;
  return s;
}

}

// rdar://12542261 stack overflow.
namespace rdar12542261 {

template <class _Tp>
struct check_complete
{
  static_assert(sizeof(_Tp) > 0, "Type must be complete.");
};


template<class _Rp>
class function // expected-note 2 {{candidate}}
{
public:
  template<class _Fp>
  function(_Fp, typename check_complete<_Fp>::type* = 0);  // expected-note {{candidate}}
};

void foobar()
{
  auto LeftCanvas = new Canvas(); // expected-error {{unknown type name}}
  function<void()> m_OnChange = [&, LeftCanvas]() { }; // expected-error {{no viable conversion}}
}

}

namespace b6981007 {
  struct S {}; // expected-note 3{{candidate}}
  void f() {
    S s(1, 2, 3); // expected-error {{no matching}}
    for (auto x : s) { // expected-error {{invalid range expression of}}
      // We used to attempt to evaluate the initializer of this variable,
      // and crash because it has an undeduced type.
      const int &n(x);
      constexpr int k = sizeof(x);
    }
  }
}

namespace incorrect_auto_type_deduction_for_typo {
struct S {
  template <typename T> S(T t) {
    (void)sizeof(t);
    (void)new auto(t);
  }
};

void Foo(S);

void test(int some_number) {  // expected-note {{'some_number' declared here}}
  auto x = sum_number;  // expected-error {{use of undeclared identifier 'sum_number'; did you mean 'some_number'?}}
  auto lambda = [x] {};
  Foo(lambda);
}
}

namespace pr29091 {
  struct X{ X(const X &x); };
  struct Y: X { using X::X; };
  bool foo() { return __has_nothrow_constructor(Y); }
  bool bar() { return __has_nothrow_copy(Y); }

  struct A { template <typename T> A(); };
  struct B : A { using A::A; };
  bool baz() { return __has_nothrow_constructor(B); }
  bool qux() { return __has_nothrow_copy(B); }
}
