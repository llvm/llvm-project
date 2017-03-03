// RUN: %clang_analyze_cc1 -I %S/Inputs -analyzer-checker=core,unix -verify %s
// expected-no-diagnostics

#include "include/sys/queue.h"

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);

int radar12491259() {
    int *p = malloc(12);
    FREE_POINTER(p);
    FREE_POINTER(p); // no-warning: we are suppressing errors coming from sys/queue macros.
    return 0;
}

#define MYMACRO(p) FREE_POINTER(p)

int radar12491259_inside_macro() {
    int *p = malloc(12);
    MYMACRO(p);
    MYMACRO(p); // no-warning: we are suppressing errors coming from sys/queue macros.
    return 0;
}
