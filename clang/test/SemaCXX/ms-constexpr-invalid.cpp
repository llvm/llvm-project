// RUN: %clang_cc1 -fms-compatibility -fms-compatibility-version=19.33 -std=c++20 -verify %s
// RUN: %clang_cc1 -fms-compatibility -fms-compatibility-version=19.33 -std=c++17 -verify %s

// Check explicitly invalid code

void runtime() {} // expected-note {{declared here}}

[[msvc::constexpr]] void f0() { runtime(); } // expected-error {{constexpr function never produces a constant expression}} \
                                             // expected-note {{non-constexpr function 'runtime' cannot be used in a constant expression}}
[[msvc::constexpr]] constexpr void f1() {} // expected-error {{attribute 'msvc::constexpr' cannot be applied to the constexpr function 'f1'}}
#if __cplusplus >= 202202L
[[msvc::constexpr]] consteval void f2() {} // expected-error {{attribute 'msvc::constexpr' cannot be applied to the consteval function 'f1'}}
#endif

struct B1 {};
struct D1 : virtual B1 { // expected-note {{virtual base class declared here}}
    [[msvc::constexpr]] D1() {} // expected-error {{constexpr constructor not allowed in struct with virtual base class}}
};

struct [[msvc::constexpr]] S2{}; // expected-error {{'constexpr' attribute only applies to functions and return statements}}

// Check invalid code mixed with valid code

[[msvc::constexpr]] int f4(int x) { return x > 1 ? 1 + f4(x / 2) : 0; } // expected-note {{non-constexpr function 'f4' cannot be used in a constant expression}} \
                                                                        // expected-note {{declared here}} \
                                                                        // expected-note {{declared here}} \
                                                                        // expected-note {{declared here}}
constexpr bool f5() { [[msvc::constexpr]] return f4(32) == 5; } // expected-note {{in call to 'f4(32)'}}
static_assert(f5()); // expected-error {{static assertion expression is not an integral constant expression}} \
                     // expected-note {{in call to 'f5()'}}

int f6(int x) { [[msvc::constexpr]] return x > 1 ? 1 + f6(x / 2) : 0; } // expected-note {{declared here}} \
                                                                        // expected-note {{declared here}}
constexpr bool f7() { [[msvc::constexpr]] return f6(32) == 5; } // expected-error {{constexpr function never produces a constant expression}} \
                                                                // expected-note {{non-constexpr function 'f6' cannot be used in a constant expression}} \
                                                                // expected-note {{non-constexpr function 'f6' cannot be used in a constant expression}}
static_assert(f7()); // expected-error {{static assertion expression is not an integral constant expression}} \
                     // expected-note {{in call to 'f7()'}}

constexpr bool f8() { // expected-error {{constexpr function never produces a constant expression}}
    [[msvc::constexpr]] f4(32); // expected-error {{'constexpr' attribute only applies to functions and return statements}} \
                                // expected-note {{non-constexpr function 'f4' cannot be used in a constant expression}} \
                                // expected-note {{non-constexpr function 'f4' cannot be used in a constant expression}}
    [[msvc::constexpr]] int i5 = f4(32); // expected-error {{'constexpr' attribute only applies to functions and return statements}}
    return i5 == 5;
}
static_assert(f8()); // expected-error {{static assertion expression is not an integral constant expression}} \
                     // expected-note {{in call to 'f8()'}}

#if __cplusplus == 201702L
struct S1 { [[msvc::constexpr]] virtual bool vm() const { return true; } }; // expected-error {{attribute 'msvc::constexpr' ignored, it only applies to function definitions and return statements}}
#endif
