// RUN: %clang_cc1 -std=c++98 %s -Wno-unused -verify
// RUN: %clang_cc1 -std=c++11 %s -Wno-unused -verify
// RUN: %clang_cc1 -std=c++2a %s -Wno-unused -verify

void use(int);

void f() {
  const int a = 1; // expected-note {{here}}

#if __cplusplus >= 201103L
  constexpr int arr[3] = {1, 2, 3}; // expected-note 2{{here}}

  struct S { int x; int f() const; };
  constexpr S s = {0}; // expected-note 3{{here}}
  constexpr S *ps = nullptr;
  S *const &psr = ps; // expected-note 2{{here}}
#endif

  struct Inner {
    void test(int i) {
      // id-expression
      use(a);

#if __cplusplus >= 201103L
      // subscripting operation with an array operand
      use(arr[i]);
      use(i[arr]);
      use((+arr)[i]); // expected-error {{reference to local variable}}
      use(i[+arr]); // expected-error {{reference to local variable}}

      // class member access naming non-static data member
      use(s.x);
      use(s.f()); // expected-error {{reference to local variable}}
      use((&s)->x); // expected-error {{reference to local variable}}
      use(ps->x); // ok (lvalue-to-rvalue conversion applied to id-expression)
      use(psr->x); // expected-error {{reference to local variable}}

      // class member access naming a static data member
      // FIXME: How to test this?

      // pointer-to-member expression
      use(s.*&S::x);
      use((s.*&S::f)()); // expected-error {{reference to local variable}}
      use(ps->*&S::x); // ok (lvalue-to-rvalue conversion applied to id-expression)
      use(psr->*&S::x); // expected-error {{reference to local variable}}
#endif

      // parentheses
      use((a));
#if __cplusplus >= 201103L
      use((s.x));
#endif

      // glvalue conditional expression
      use(i ? a : a);
      use(i ? i : a);

      // comma expression
      use((i, a));
      // FIXME: This is not an odr-use because it is a discarded-value
      // expression applied to an expression whose potential result is 'a'.
      use((a, a)); // expected-error {{reference to local variable}}

      // (and combinations thereof)
      use(a ? (i, a) : a);
#if __cplusplus >= 201103L
      use(a ? (i, a) : arr[a ? s.x : arr[a]]);
#endif
    }
  };
}

// FIXME: Test that this behaves properly.
namespace std_example {
  struct S { static const int x = 0, y = 0; };
  const int &f(const int &r);
  bool b;
  int n = b ? (1, S::x)
            : f(S::y);
}
