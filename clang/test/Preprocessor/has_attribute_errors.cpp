// RUN: %clang_cc1 -triple i386-unknown-unknown -Eonly -verify %s

// We warn users if they write an attribute like
// [[__clang__::fallthrough]] because __clang__ is a macro that expands to 1.
// Instead, we suggest users use [[_Clang::fallthrough]] in this situation.
// However, because __has_cpp_attribute (and __has_c_attribute) require
// expanding their argument tokens, __clang__ expands to 1 in the feature test
// macro as well. We don't currently give users a kind warning in this case,
// but we previously did not expand macros and so this would return 0. Now that
// we properly expand macros, users will now get an error about using incorrect
// syntax.

__has_cpp_attribute(__clang__::fallthrough) // expected-error {{missing ')' after <numeric_constant>}} \
                                            // expected-note {{to match this '('}} \
                                            // expected-error {{builtin feature check macro requires a parenthesized identifier}}

