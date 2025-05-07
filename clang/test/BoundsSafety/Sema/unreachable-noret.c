

// RUN: %clang_cc1 -fbounds-safety -Wunreachable-code -verify %s
// RUN: %clang_cc1 -fbounds-safety -Wunreachable-code -x objective-c -fexperimental-bounds-safety-objc -verify %s

// expected-no-diagnostics

#include <ptrcheck.h>

#define NO_RETURN __attribute__((noreturn))
void NO_RETURN halt(const void * const p_fatal_error);
static void NO_RETURN handler_private(const void *__sized_by(0x78) p_stack)
{
  int foo;
  halt(&foo);
}

void NO_RETURN handler_irq(const void *__sized_by(0x78) p_stack)
{
  handler_private(p_stack);
}
