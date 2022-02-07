// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only \
// RUN:            -Wtautological-unsigned-enum-zero-compare \
// RUN:            -verify=unsigned,unsigned-signed %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only \
// RUN:            -Wtautological-unsigned-enum-zero-compare \
// RUN:            -verify=unsigned-signed %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only \
// RUN:            -verify=silence %s

// Okay, this is where it gets complicated.
// Then default enum sigdness is target-specific.
// On windows, it is signed by default. We do not want to warn in that case.

int main(void) {
  enum A { A_a = 0 };
  enum A a;
  enum B { B_a = -1 };
  enum B b;

  // silence-no-diagnostics

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

  if (b < 0)
    return 0;
  if (0 >= b)
    return 0;
  if (b > 0)
    return 0;
  if (0 <= b)
    return 0;
  if (b <= 0)
    return 0;
  if (0 > b)
    return 0;
  if (b >= 0)
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

  if (a == 0)
    return 0;
  if (0 != a)
    return 0;
  if (a != 0)
    return 0;
  if (0 == a)
    return 0;

  if (a == 0U)
    return 0;
  if (0U != a)
    return 0;
  if (a != 0U)
    return 0;
  if (0U == a)
    return 0;

  if (b == 0)
    return 0;
  if (0 != b)
    return 0;
  if (b != 0)
    return 0;
  if (0 == b)
    return 0;

  if (b == 0U)
    return 0;
  if (0U != b)
    return 0;
  if (b != 0U)
    return 0;
  if (0U == b)
    return 0;

  return 1;
}
