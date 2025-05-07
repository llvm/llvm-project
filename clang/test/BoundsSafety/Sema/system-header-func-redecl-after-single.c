
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s -verify-ignore-unexpected=note
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s -verify-ignore-unexpected=note

// expected-no-diagnostics

#include <ptrcheck.h>
#include "system-header-func-decl.h"

int *__single foo(int *__single *__single);

void bar(void) {
    int *__single s;
    foo(&s);
}
