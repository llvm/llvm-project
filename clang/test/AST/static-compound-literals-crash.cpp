// FIXME: The temporary bound to the reference member is not lifetime-extended
// like it would be for a static variable, so the initializer is rejected.
// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s
namespace case1 {
struct RR { int&& r; };
struct Z { RR* x; };
constinit Z z = { (RR[1]){1} }; // expected-error {{initializer element is not a compile-time constant}}
}
