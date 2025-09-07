// RUN: mkdir -p %t.dir && cd %t.dir
// RUN: %clang_tsan -O1 %s -DBUILD_LIB=1 -fno-sanitize=thread -shared -fPIC -o %dynamiclib %ld_flags_rpath_so
// RUN: %clang_tsan -O1 %s -o %t %ld_flags_rpath_exe
// RUN: %run %t | FileCheck %s

// Test that initialization/finalization hooks are called, even when they are
// not defined in the main executable, but by another another library that
// doesn't directly link against the TSan runtime.

#include <stdio.h>

#if BUILD_LIB

extern "C" void __tsan_on_initialize() {
  printf("__tsan_on_initialize()\n");
}

extern "C" int __tsan_on_finalize(int failed) {
  printf("__tsan_on_finalize()\n");
  return failed;
}

#else // BUILD_LIB

int main() {
  printf("main()\n");
  return 0;
}

#endif // BUILD_LIB

// CHECK: __tsan_on_initialize()
// CHECK: main()
// CHECK: __tsan_on_finalize()
