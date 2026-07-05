// RUN: %clangxx -O0 %s -o %t && %run %t 2>&1 | FileCheck %s

// Test that symbolizer does not crash on frame with large function name.

// On Darwin LSan reports a false positive
// UNSUPPORTED: darwin && lsan

#include <sanitizer/common_interface_defs.h>
#include <vector>

template <int N> struct A {
  template <class T> void RecursiveTemplateFunction(const T &t);
};

template <int N>
template <class T>
__attribute__((noinline)) void A<N>::RecursiveTemplateFunction(const T &) {
  std::vector<T> t;
  return A<N - 1>().RecursiveTemplateFunction(t);
}

template <>
template <class T>
__attribute__((noinline)) void A<0>::RecursiveTemplateFunction(const T &) {
  __sanitizer_print_stack_trace();
}

int main() {
  // CHECK: {{vector<.*vector<.*vector<.*vector<.*vector<}}
  A<10>().RecursiveTemplateFunction(0);
}
