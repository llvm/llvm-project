
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void foo(int *__counted_by(count) p, int count) {
	int *end = p + 9;
  p = end; // expected-note{{'p' has been assigned here}}
  count = (int)(end - p); // expected-error{{cannot reference 'p' after it is changed during consecutive assignments}}
}
