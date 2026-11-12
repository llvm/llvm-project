// RUN: %clang_cc1 -std=c++1y %s -verify

static_assert(0b1001 == 9, "");

using I = int;
using I = decltype(0b101001);
using ULL = unsigned long long;
using ULL = decltype(0b10101001ULL);

constexpr unsigned long long operator""_foo(unsigned long long n) {
  return n * 2;
}
static_assert(0b10001111_foo == 286, "");

int k1 = 0b1234; // expected-error {{invalid digit '2' in binary constant}}
// FIXME: If we ever need to support a standard suffix starting with [a-f],
// we'll need to rework our binary literal parsing rules.
int k2 = 0b10010f; // expected-error {{invalid digit 'f' in binary constant}}
int k3 = 0b10010g; // expected-error {{invalid suffix 'g' on integer constant}}
int k4 = 0b; // expected-error {{invalid digit 'b' in octal constant}}
