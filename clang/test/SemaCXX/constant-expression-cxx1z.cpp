// RUN: %clang_cc1 -std=c++1z -verify %s -fcxx-exceptions -triple=x86_64-linux-gnu

namespace BaseClassAggregateInit {
  struct A {
    int a, b, c;
    constexpr A(int n) : a(n), b(3 * n), c(b - 1) {} // expected-note {{outside the range of representable}}
    constexpr A() : A(10) {};
  };
  struct B : A {};
  struct C { int q; };
  struct D : B, C { int k; };

  constexpr D d1 = { 1, 2, 3 };
  static_assert(d1.a == 1 && d1.b == 3 && d1.c == 2 && d1.q == 2 && d1.k == 3);

  constexpr D d2 = { 14 };
  static_assert(d2.a == 14 && d2.b == 42 && d2.c == 41 && d2.q == 0 && d2.k == 0);

  constexpr D d3 = { A(5), C{2}, 1 };
  static_assert(d3.a == 5 && d3.b == 15 && d3.c == 14 && d3.q == 2 && d3.k == 1);

  constexpr D d4 = {};
  static_assert(d4.a == 10 && d4.b == 30 && d4.c == 29 && d4.q == 0 && d4.k == 0);

  constexpr D d5 = { __INT_MAX__ }; // expected-error {{must be initialized by a constant expression}}
  // expected-note-re@-1 {{in call to 'A({{.*}})'}}
}

namespace NoexceptFunctionTypes {
  template<typename T> constexpr bool f() noexcept(true) { return true; }
  constexpr bool (*fp)() = f<int>;
  static_assert(f<int>());
  static_assert(fp());

  template<typename T> struct A {
    constexpr bool f() noexcept(true) { return true; }
    constexpr bool g() { return f(); }
    constexpr bool operator()() const noexcept(true) { return true; }
  };
  static_assert(A<int>().f());
  static_assert(A<int>().g());
  static_assert(A<int>()());
}

namespace Cxx17CD_NB_GB19 {
  const int &r = 0;
  constexpr int n = r;
}

namespace PR37585 {
template <class T> struct S { static constexpr bool value = true; };
template <class T> constexpr bool f() { return true; }
template <class T> constexpr bool v = true;

void test() {
  if constexpr (true) {}
  else if constexpr (f<int>()) {}
  else if constexpr (S<int>::value) {}
  else if constexpr (v<int>) {}
}
}

// Check that assignment operators evaluate their operands right-to-left.
namespace EvalOrder {
  template<typename T> struct lvalue {
    T t;
    constexpr T &get() { return t; }
  };

  struct UserDefined {
    int n = 0;
    constexpr UserDefined &operator=(const UserDefined&) { return *this; }
    constexpr UserDefined &operator+=(const UserDefined&) { return *this; }
    constexpr void operator<<(const UserDefined&) const {}
    constexpr void operator>>(const UserDefined&) const {}
    constexpr void operator+(const UserDefined&) const {}
    constexpr void operator[](int) const {}
  };
  constexpr UserDefined ud;

  struct NonMember {};
  constexpr void operator+=(NonMember, NonMember) {}
  constexpr void operator<<(NonMember, NonMember) {}
  constexpr void operator>>(NonMember, NonMember) {}
  constexpr void operator+(NonMember, NonMember) {}
  constexpr NonMember nm;

  constexpr void f(...) {}

  // Helper to ensure that 'a' is evaluated before 'b'.
  struct seq_checker {
    bool done_a = false;
    bool done_b = false;

    template <typename T> constexpr T &&a(T &&v) {
      done_a = true;
      return (T &&)v;
    }
    template <typename T> constexpr T &&b(T &&v) {
      if (!done_a)
        throw "wrong";
      done_b = true;
      return (T &&)v;
    }

    constexpr bool ok() { return done_a && done_b; }
  };

  // SEQ(expr), where part of the expression is tagged A(...) and part is
  // tagged B(...), checks that A is evaluated before B.
  #define A sc.a
  #define B sc.b
  #define SEQ(...) static_assert([](seq_checker sc) { void(__VA_ARGS__); return sc.ok(); }({}))

  // Longstanding sequencing rules.
  SEQ((A(1), B(2)));
  SEQ((A(true) ? B(2) : throw "huh?"));
  SEQ((A(false) ? throw "huh?" : B(2)));
  SEQ(A(true) && B(true));
  SEQ(A(false) || B(true));

  // From P0145R3:

  // Rules 1 and 2 have no effect ('b' is not an expression).

  // Rule 3: a->*b
  SEQ(A(ud).*B(&UserDefined::n));
  SEQ(A(&ud)->*B(&UserDefined::n));

  // Rule 4: a(b1, b2, b3)
  SEQ(A(f)(B(1), B(2), B(3)));

  // Rule 5: b = a, b @= a
  SEQ(B(lvalue<int>().get()) = A(0));
  SEQ(B(lvalue<UserDefined>().get()) = A(ud));
  SEQ(B(lvalue<int>().get()) += A(0));
  SEQ(B(lvalue<UserDefined>().get()) += A(ud));
  SEQ(B(lvalue<NonMember>().get()) += A(nm));

  // Rule 6: a[b]
  constexpr int arr[3] = {};
  SEQ(A(arr)[B(0)]);
  SEQ(A(+arr)[B(0)]);
  SEQ(A(0)[B(arr)]);
  SEQ(A(0)[B(+arr)]);
  SEQ(A(ud)[B(0)]);

  // Rule 7: a << b
  SEQ(A(1) << B(2));
  SEQ(A(ud) << B(ud));
  SEQ(A(nm) << B(nm));

  // Rule 8: a >> b
  SEQ(A(1) >> B(2));
  SEQ(A(ud) >> B(ud));
  SEQ(A(nm) >> B(nm));

  // No particular order of evaluation is specified in other cases, but we in
  // practice evaluate left-to-right.
  // FIXME: Technically we're expected to check for undefined behavior due to
  // unsequenced read and modification and treat it as non-constant due to UB.
  SEQ(A(1) + B(2));
  SEQ(A(ud) + B(ud));
  SEQ(A(nm) + B(nm));
  SEQ(f(A(1), B(2)));

  #undef SEQ
  #undef A
  #undef B
}

namespace LambdaCallOp {
  constexpr void get_lambda(void (*&p)()) { p = []{}; }
  constexpr void call_lambda() {
    void (*p)() = nullptr;
    get_lambda(p);
    p();
  }
}
