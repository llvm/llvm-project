
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s -verify-ignore-unexpected=note
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fbounds-attributes-objc-experimental -verify %s -verify-ignore-unexpected=note

// expected-no-diagnostics

#include <ptrcheck.h>
#include "system-header-func-decl.h"

int *__unsafe_indexable foo(int *__unsafe_indexable *__unsafe_indexable);

void bar(void) {
    int *__unsafe_indexable s;
    foo(&s);
}