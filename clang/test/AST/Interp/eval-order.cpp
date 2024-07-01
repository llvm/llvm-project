// RUN: %clang_cc1 -std=c++1z -verify=ref,both %s -fcxx-exceptions -triple=x86_64-linux-gnu
// RUN: %clang_cc1 -std=c++1z -verify=expected,both %s -fcxx-exceptions -triple=x86_64-linux-gnu -fexperimental-new-constant-interpreter

// ref-no-diagnostics

/// Check that assignment operators evaluate their operands right-to-left.
/// Copied from test/SemaCXX/constant-expression-cxx1z.cpp
///
/// As you can see from the FIXME comments, some of these are not yet working correctly
/// in the new interpreter.
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
        throw "wrong"; // expected-note 7{{not valid}}
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
  SEQ(A(f)(B(1), B(2), B(3))); // expected-error {{not an integral constant expression}} FIXME \
                               // expected-note 2{{in call to}}

  // Rule 5: b = a, b @= a
  SEQ(B(lvalue<int>().get()) = A(0)); // expected-error {{not an integral constant expression}} FIXME \
                                      // expected-note 2{{in call to}}
  SEQ(B(lvalue<UserDefined>().get()) = A(ud)); // expected-error {{not an integral constant expression}} FIXME \
                                               // expected-note 2{{in call to}}
  SEQ(B(lvalue<int>().get()) += A(0));
  SEQ(B(lvalue<UserDefined>().get()) += A(ud)); // expected-error {{not an integral constant expression}} FIXME \
                                                // expected-note 2{{in call to}}

  SEQ(B(lvalue<NonMember>().get()) += A(nm)); // expected-error {{not an integral constant expression}} FIXME \
                                              // expected-note 2{{in call to}}


  // Rule 6: a[b]
  constexpr int arr[3] = {};
  SEQ(A(arr)[B(0)]);
  SEQ(A(+arr)[B(0)]);
  SEQ(A(0)[B(arr)]); // expected-error {{not an integral constant expression}} FIXME \
                     // expected-note 2{{in call to}}
  SEQ(A(0)[B(+arr)]); // expected-error {{not an integral constant expression}} FIXME \
                      // expected-note 2{{in call to}}
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
