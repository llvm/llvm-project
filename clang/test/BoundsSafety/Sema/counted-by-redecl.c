

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -verify %s

#include <ptrcheck.h>

int len_a;
// expected-note@+1{{'a' declared here}}
int * __counted_by(len_a) a;
// expected-error@+1{{conflicting '__counted_by_or_null' attribute with the previous variable declaration}}
int * __counted_by_or_null(len_a) a;

int len_b;
// expected-note@+1{{'b' declared here}}
int * __counted_by_or_null(len_b) b;
// expected-error@+1{{conflicting '__counted_by' attribute with the previous variable declaration}}
int * __counted_by(len_b) b;

int len_c;
// expected-note@+1{{'c' declared here}}
int * __sized_by(len_c) c;
// expected-error@+1{{conflicting '__sized_by_or_null' attribute with the previous variable declaration}}
int * __sized_by_or_null(len_c) c;

int len_d;
// expected-note@+1{{'d' declared here}}
int * __sized_by_or_null(len_d) d;
// expected-error@+1{{conflicting '__sized_by' attribute with the previous variable declaration}}
int * __sized_by(len_d) d;
