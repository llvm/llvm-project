// The temporary bound to the reference member is not lifetime-extended by the
// file-scope compound literal, so the element is not a constant initializer.
// FIXME: Extend the temporary's lifetime to that of the compound literal.
// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s
namespace case1 {
struct RR { int&& r; };
struct Z { RR* x; };
constinit Z z = { (RR[1]){1} }; // expected-error {{initializer element is not a compile-time constant}}
}
