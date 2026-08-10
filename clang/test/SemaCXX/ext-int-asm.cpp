// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wimplicit-int-conversion -triple x86_64-gnu-linux -fasm-blocks

void NotAllowedInInlineAsm(_BitInt(9) in, _BitInt(9) out) {
  __asm { mov eax, in} // expected-error{{invalid type '_BitInt(9)' in asm input}}
  __asm { mov out, eax} // expected-error{{invalid type '_BitInt(9)' in asm output}}

  asm("" : "=g" (in));// expected-error{{invalid type '_BitInt(9)' in asm input}}
  asm("" :: "r" (out));// expected-error{{invalid type '_BitInt(9)' in asm output}}

}
