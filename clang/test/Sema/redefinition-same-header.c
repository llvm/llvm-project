// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: echo 'int yyy = 42;' > %t/a.h
// RUN: %clang_cc1 -fsyntax-only %s -I%t  -verify

// expected-error@a.h:1 {{redefinition of 'yyy'}}
// expected-note@a.h:1 {{unguarded header; consider using #ifdef guards or #pragma once}}
// expected-note-re@redefinition-same-header.c:11 {{'{{.*}}a.h' included multiple times, additional include site here}}
// expected-note-re@redefinition-same-header.c:12 {{'{{.*}}a.h' included multiple times, additional include site here}}

#include "a.h"
#include "a.h"

int foo() { return yyy; }
