// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: echo 'int yyy = 42;' > %t/a.h
// RUN: %clang_cc1 -fsyntax-only %s -I%t  -verify

// expected-error@a.h:1 {{redefinition of 'yyy'}}
// expected-note-re@redefinition-same-header.c:9 {{'{{.*}}/a.h' included multiple times, consider augmenting this header with #ifdef guards}}

#include "a.h"
#include "a.h"

int foo() { return yyy; }
