
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: not %clang_cc1 -fsyntax-only -fbounds-safety  -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: not %clang_cc1 -fsyntax-only -fbounds-safety  -x objective-c -fexperimental-bounds-safety-objc -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

int len;

// Using attribute directly should not suggest a fix-it.
// expected-error-re@+1{{cannot apply '__counted_by' attribute to 'void *' because 'void' has unknown size{{$}}}}
void *__attribute__((__counted_by__(len))) voidPtr;

// Using a custom macro that wraps the attribute should not suggest a fix-it.
// expected-error-re@+2{{cannot apply '__counted_by' attribute to 'void *' because 'void' has unknown size{{$}}}}
#define my_custom_counted_by(X) __attribute__((__counted_by__(X)))
void * my_custom_counted_by(len) voidPtr2;

// CHECK-NOT: fix-it:"{{.+}}":
