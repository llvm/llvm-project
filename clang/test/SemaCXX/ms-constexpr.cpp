// RUN: %clang_cc1 -fms-compatibility -fms-compatibility-version=19.33 -std=c++20 -verify %s

[[msvc::constexpr]] int log2(int x) { [[msvc::constexpr]] return x > 1 ? 1 + log2(x / 2) : 0; }
constexpr bool test_log2() { [[msvc::constexpr]] return log2(32) == 5; }
static_assert(test_log2());

[[msvc::constexpr]] int get_value(int x)
{
  switch (x)
  {
    case 42: return 1337;
    default:
             if (x < 0) [[msvc::constexpr]] return log2(-x);
             else return x;
  }
}

constexpr bool test_complex_expr() {
  [[msvc::constexpr]] return get_value(get_value(42) - 1337 + get_value(-32) - 5 + (get_value(1) ? get_value(0) : get_value(2))) == get_value(0);
}
static_assert(test_complex_expr());

constexpr bool get_constexpr_true() { return true; }
[[msvc::constexpr]] bool get_msconstexpr_true() { return get_constexpr_true(); }
constexpr bool test_get_msconstexpr_true() { [[msvc::constexpr]] return get_msconstexpr_true(); }
static_assert(test_get_msconstexpr_true());

// TODO (#72149): Add support for [[msvc::constexpr]] constructor; this code is valid for MSVC.
struct S2 {
    [[msvc::constexpr]] S2() {}
    [[msvc::constexpr]] bool value() { return true; }
    static constexpr bool check() { [[msvc::constexpr]] return S2{}.value(); } // expected-error {{constexpr function never produces a constant expression}} \
                                                                               // expected-note {{non-literal type 'S2' cannot be used in a constant expression}} \
                                                                               // expected-note {{non-literal type 'S2' cannot be used in a constant expression}}
};
static_assert(S2::check()); // expected-error {{static assertion expression is not an integral constant expression}} \
                            // expected-note {{in call to 'check()'}}
