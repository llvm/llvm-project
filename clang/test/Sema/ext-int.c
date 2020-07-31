// RUN: %clang_cc1 -fsyntax-only -verify %s -Wimplicit-int-conversion -triple x86_64-gnu-linux

typedef _ExtInt(31) EI31;

void Ternary(_ExtInt(30) s30, EI31 s31a, _ExtInt(31) s31b,
             _ExtInt(32) s32, int b) {
  b ? s30 : s31a; // expected-error{{incompatible operand types}}
  b ? s31a : s30; // expected-error{{incompatible operand types}}
  b ? s32 : 0; // expected-error{{incompatible operand types}}
  (void)(b ? s31a : s31b);
  (void)(s30 ? s31a : s31b);
}
