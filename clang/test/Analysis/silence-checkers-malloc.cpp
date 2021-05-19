// RUN: %clang_analyze_cc1 -verify="no-silence" %s \
// RUN:   -triple i386-unknown-linux-gnu \
// RUN:   -analyzer-checker=core,apiModeling \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=cplusplus.NewDelete

// RUN: %clang_analyze_cc1 -verify="unix-silenced" %s \
// RUN:   -triple i386-unknown-linux-gnu \
// RUN:   -analyzer-checker=core,apiModeling \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=cplusplus.NewDelete\
// RUN:   -analyzer-config silence-checkers="unix"

#include "Inputs/system-header-simulator-cxx.h"

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);
char *strdup(const char *s);

void checkThatMallocCheckerIsRunning() {
  malloc(4);
} // no-silence-warning{{Potential memory leak [unix.Malloc]}}

int const_ptr_and_callback_def_param_null(int, const char *, int n, void (*)(void *) = 0);
void r11160612_no_callback() {
  char *x = (char *)malloc(12);
  const_ptr_and_callback_def_param_null(0, x, 12);
} // no-silence-warning{{Potential leak of memory pointed to by 'x' [unix.Malloc]}}

#define ZERO_SIZE_PTR ((void *)16)

void test_delete_ZERO_SIZE_PTR() {
  int *Ptr = (int *)ZERO_SIZE_PTR;
  // ZERO_SIZE_PTR is specially handled but only for malloc family
  delete Ptr; // no-silence-warning{{Argument to 'delete' is a constant address (16), which is not memory allocated by 'new' [cplusplus.NewDelete]}}
              // unix-silenced-warning@-1{{Argument to 'delete' is a constant address (16), which is not memory allocated by 'new' [cplusplus.NewDelete]}}
}
