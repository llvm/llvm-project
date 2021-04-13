// RUN: %clang_cc1 -fsyntax-only \
// RUN:            -fno-signed-char \
// RUN:            -Wtautological-unsigned-zero-compare \
// RUN:            -Wtautological-unsigned-char-zero-compare \
// RUN:            -verify=unsigned %s
// RUN: %clang_cc1 -fsyntax-only \
// RUN:            -Wtautological-unsigned-zero-compare \
// RUN:            -Wtautological-unsigned-char-zero-compare \
// RUN:            -verify=signed %s

void f(char c, unsigned char uc, signed char cc) {
  if (c < 0)
    return;
  // unsigned-warning@-2 {{comparison of char expression < 0 is always false, since char is interpreted as unsigned}}
  if (uc < 0)
    return;
  // unsigned-warning@-2 {{comparison of unsigned expression < 0 is always false}}
  // signed-warning@-3 {{comparison of unsigned expression < 0 is always false}}
  if (cc < 0)
    return;
  // Promoted to integer expressions should not warn.
  if (c - 4 < 0)
    return;
}

void ref(char &c, unsigned char &uc, signed char &cc) {
  if (c < 0)
    return;
  // unsigned-warning@-2 {{comparison of char expression < 0 is always false, since char is interpreted as unsigned}}
  if (uc < 0)
    return;
  // unsigned-warning@-2 {{comparison of unsigned expression < 0 is always false}}
  // signed-warning@-3 {{comparison of unsigned expression < 0 is always false}}
  if (cc < 0)
    return;
  // Promoted to integer expressions should not warn.
  if (c - 4 < 0)
    return;
}
