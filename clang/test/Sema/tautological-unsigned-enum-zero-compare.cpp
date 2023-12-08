// RUN: %clang_cc1 -std=c++11 -triple=x86_64-pc-linux-gnu -fsyntax-only \
// RUN:            -Wtautological-unsigned-enum-zero-compare \
// RUN:            -verify=unsigned,unsigned-signed %s
// RUN: %clang_cc1 -std=c++11 -triple=x86_64-pc-win32 -fsyntax-only \
// RUN:            -Wtautological-unsigned-enum-zero-compare \
// RUN:            -verify=unsigned-signed %s
// RUN: %clang_cc1 -std=c++11 -triple=x86_64-pc-win32 -fsyntax-only \
// RUN:            -verify=silence %s

// silence-no-diagnostics

int main() {
  // On Windows, all enumerations have a fixed underlying type, which is 'int'
  // if not otherwise specified, so A is identical to C on Windows. Otherwise,
  // we follow the C++ rules, which say that the only valid values of A are 0
  // and 1.
  enum A { A_foo = 0, A_bar, };
  enum A a;

  enum B : unsigned { B_foo = 0, B_bar, };
  enum B b;

  enum C : signed { C_foo = 0, C_bar, };
  enum C c;

  if (a < 0) // unsigned-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0 >= a)
    return 0;
  if (a > 0)
    return 0;
  if (0 <= a) // unsigned-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (a <= 0)
    return 0;
  if (0 > a) // unsigned-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (a >= 0) // unsigned-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0 < a)
    return 0;

  // FIXME: As below, the issue here is that the enumeration is promoted to
  // unsigned.
  if (a < 0U) // unsigned-signed-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0U >= a)
    return 0;
  if (a > 0U)
    return 0;
  if (0U <= a) // unsigned-signed-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (a <= 0U)
    return 0;
  if (0U > a) // unsigned-signed-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (a >= 0U) // unsigned-signed-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0U < a)
    return 0;

  if (b < 0) // unsigned-signed-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0 >= b)
    return 0;
  if (b > 0)
    return 0;
  if (0 <= b) // unsigned-signed-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (b <= 0)
    return 0;
  if (0 > b) // unsigned-signed-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (b >= 0) // unsigned-signed-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0 < b)
    return 0;

  if (b < 0U) // unsigned-signed-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0U >= b)
    return 0;
  if (b > 0U)
    return 0;
  if (0U <= b) // unsigned-signed-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (b <= 0U)
    return 0;
  if (0U > b) // unsigned-signed-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (b >= 0U) // unsigned-signed-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0U < b)
    return 0;

  if (c < 0)
    return 0;
  if (0 >= c)
    return 0;
  if (c > 0)
    return 0;
  if (0 <= c)
    return 0;
  if (c <= 0)
    return 0;
  if (0 > c)
    return 0;
  if (c >= 0)
    return 0;
  if (0 < c)
    return 0;

  // FIXME: These diagnostics are terrible. The issue here is that the signed
  // enumeration value was promoted to an unsigned type.
  if (c < 0U) // unsigned-signed-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;
  if (0U >= c)
    return 0;
  if (c > 0U)
    return 0;
  if (0U <= c) // unsigned-signed-warning {{comparison of 0 <= unsigned enum expression is always true}}
    return 0;
  if (c <= 0U)
    return 0;
  if (0U > c) // unsigned-signed-warning {{comparison of 0 > unsigned enum expression is always false}}
    return 0;
  if (c >= 0U) // unsigned-signed-warning {{comparison of unsigned enum expression >= 0 is always true}}
    return 0;
  if (0U < c)
    return 0;

  return 1;
}

namespace crash_enum_zero_width {
int test() {
  enum A : unsigned {
    A_foo = 0
  };
  enum A a;

  // used to crash in llvm::APSInt::getMaxValue()
  if (a < 0) // unsigned-signed-warning {{comparison of unsigned enum expression < 0 is always false}}
    return 0;

  return 1;
}
} // namespace crash_enum_zero_width
