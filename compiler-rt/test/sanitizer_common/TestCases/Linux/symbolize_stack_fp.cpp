/// When the main program doesn't use .eh_frame, the slow unwinder does not work.
/// Test that we can fall back to the fast unwinder.
// RUN: %clangxx -O0 -g1 -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer %s -o %t
// RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SEC
// RUN: %run %t 2>&1 | FileCheck %s

// On android %t is a wrapper python script so llvm-readelf will fail.
// UNSUPPORTED: android

/// No .eh_frame && -g => .debug_frame
// SEC: .debug_frame

#include <sanitizer/common_interface_defs.h>
#include <vector>

template <int N>
struct A {
  template <class T>
  void RecursiveTemplateFunction(const T &t);
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
  A<7>().RecursiveTemplateFunction(0);
}
