// RUN: %clang_cc1 %s -triple powerpc64-unknown-linux-gnu -fasm-blocks -verify -fsyntax-only

void f(void) {
  __asm nop // expected-error {{unsupported architecture 'powerpc64' for MS-style inline assembly}}
}
