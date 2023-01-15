// RUN: %clang_cc1 %s -ffreestanding -triple=i386-unknown-unknown \
// RUN: -target-feature +cmpccxadd  -emit-llvm -fsyntax-only -verify

#include <immintrin.h>

int test_cmpccxadd32(void *__A, int __B, int __C) {
  return _cmpccxadd_epi32(__A, __B, __C, 0); // expected-error {{call to undeclared function '_cmpccxadd_epi32'}}
}
