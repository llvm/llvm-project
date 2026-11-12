// RUN: %clang_cc1 -DTEMPLATE_OVERLOAD -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

#include <stddef.h>

// Note that each test must be run separately so it can be the first operator
// new declaration in the file.

#if defined(TEMPLATE_OVERLOAD)
// Don't crash on global template operator new overloads.
template<typename T> void* operator new(size_t, T);
void test_template_overload() {
  (void)new(0) double;
}
#endif

void test_predefined() {
  (void)new double;
}
