
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fbounds-attributes-objc-experimental -verify %s

#include <ptrcheck.h>

struct type {
	int count;
	char [__counted_by(count)]; // expected-error{{expected member name or ';' after declaration specifiers}}
};
