

// Use arm64-apple-ios to ensure that sizeof(uint8_t) < sizeof(uintptr_t).
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>
#include <stdint.h>

char buf[42];

// ok
uintptr_t buf_ptr = (uintptr_t)buf;

// expected-error@+2{{initializer element is not a compile-time constant}}
// expected-warning@+1{{cast to smaller integer type 'uint8_t' (aka 'unsigned char') from 'char *__bidi_indexable'}}
uint8_t buf_8 = (uint8_t)buf;

void foo(int *__single s, int *__indexable i, int *__bidi_indexable bi) {
  uintptr_t s_ptr = (uintptr_t)s;    // ok
  uintptr_t i_ptr = (uintptr_t)i;    // ok
  uintptr_t bi_ptr = (uintptr_t)bi;  // ok

  uint8_t s_8 = (uint8_t)s;    // expected-warning{{cast to smaller integer type 'uint8_t' (aka 'unsigned char') from 'int *__single'}}
  uint8_t i_8 = (uint8_t)i;    // expected-warning{{cast to smaller integer type 'uint8_t' (aka 'unsigned char') from 'int *__indexable'}}
  uint8_t bi_8 = (uint8_t)bi;  // expected-warning{{cast to smaller integer type 'uint8_t' (aka 'unsigned char') from 'int *__bidi_indexable'}}
}
