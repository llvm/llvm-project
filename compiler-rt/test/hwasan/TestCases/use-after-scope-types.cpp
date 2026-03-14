// This is the ASAN test of the same name ported to HWAsan.

// RUN: %clangxx_hwasan -std=c++11 -O0 %s -o %t
// RUN: %clangxx_hwasan -fno-exceptions -std=c++11 -O0 %s -o %t-noexcept

// RUN: not %run %t 0 2>&1 | FileCheck %s
// RUN: not %run %t 1 2>&1 | FileCheck %s
// RUN: not %run %t 2 2>&1 | FileCheck %s
// RUN: not %run %t 3 2>&1 | FileCheck %s
// RUN: not %run %t 4 2>&1 | FileCheck %s
// RUN: not %run %t 5 2>&1 | FileCheck %s
// RUN: not %run %t 6 2>&1 | FileCheck %s
// RUN: not %run %t 7 2>&1 | FileCheck %s
// RUN: not %run %t 8 2>&1 | FileCheck %s
// RUN: not %run %t 9 2>&1 | FileCheck %s
// RUN: not %run %t 10 2>&1 | FileCheck %s

// RUN: not %run %t-noexcept 0 2>&1 | FileCheck %s
// RUN: not %run %t-noexcept 1 2>&1 | FileCheck %s
// RUN: not %run %t-noexcept 2 2>&1 | FileCheck %s
// RUN: not %run %t-noexcept 3 2>&1 | FileCheck %s
// RUN: not %run %t-noexcept 4 2>&1 | FileCheck %s
// RUN: not %run %t-noexcept 5 2>&1 | FileCheck %s
// RUN: not %run %t-noexcept 6 2>&1 | FileCheck %s
// RUN: not %run %t-noexcept 7 2>&1 | FileCheck %s
// RUN: not %run %t-noexcept 8 2>&1 | FileCheck %s
// RUN: not %run %t-noexcept 9 2>&1 | FileCheck %s
// RUN: not %run %t-noexcept 10 2>&1 | FileCheck %s

// REQUIRES: aarch64-target-arch || riscv64-target-arch

#include <stdlib.h>
#include <string>
#include <vector>

template <class T>
struct Ptr {
  void Store(T *ptr) { t = ptr; }

  void Access() { *t = {}; }

  T *t;
};

template <class T, size_t N>
struct Ptr<T[N]> {
  using Type = T[N];
  void Store(Type *ptr) { t = *ptr; }

  void Access() { *t = {}; }

  T *t;
};

template <class T>
__attribute__((noinline)) void test() {
  Ptr<T> ptr;
  {
    T x;
    ptr.Store(&x);
  }

  ptr.Access();
  // CHECK: ERROR: HWAddressSanitizer: tag-mismatch
  // CHECK:  #{{[0-9]+}} 0x{{.*}} in {{(void )?test.*\((void)?\) .*}}use-after-scope-types.cpp
  // CHECK: Cause: stack tag-mismatch
}

int main(int argc, char **argv) {
  using Tests = void (*)();
  Tests tests[] = {
      &test<bool>,
      &test<char>,
      &test<int>,
      &test<double>,
      &test<float>,
      &test<void *>,
      &test<std::vector<std::string>>,
      &test<int[3]>,
      &test<int[1000]>,
      &test<char[3]>,
      &test<char[1000]>,
  };

  int n = atoi(argv[1]);
  if (n == sizeof(tests) / sizeof(tests[0])) {
    for (auto te : tests)
      te();
  } else {
    tests[n]();
  }

  return 0;
}
