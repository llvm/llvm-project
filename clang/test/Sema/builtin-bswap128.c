// RUN: %clang_cc1 -triple i686-linux-gnu -fsyntax-only -verify %s

void test(void) {
  __builtin_bswap128(1); // expected-error {{use of unknown builtin '__builtin_bswap128'}}
}
