// RUN: %clang_cc1 -fsyntax-only -verify -ffreestanding -triple x86_64-linux-pc %s

// expected-error@+1 3 {{unknown type name '__bfloat16'}}
__bfloat16 foo(__bfloat16 a, __bfloat16 b) {
  return a + b;
}

#include <immintrin.h>

// expected-warning@+2 3 {{'__bfloat16' is deprecated: use __bf16 instead}}
// expected-note@* 3 {{'__bfloat16' has been explicitly marked deprecated here}}
__bfloat16 bar(__bfloat16 a, __bfloat16 b) {
  return a + b;
}
