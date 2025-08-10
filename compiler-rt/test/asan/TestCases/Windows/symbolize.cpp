// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=symbolize=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SYMBOLIZE-OFF
// RUN: %env_asan_opts=symbolize=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SYMBOLIZE-ON

// RUN: %clangxx_asan -O0 %s -o %t -DUSER_FUNCTION_OFF
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SYMBOLIZE-OFF
// RUN: %env_asan_opts=symbolize=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SYMBOLIZE-OFF
// RUN: %env_asan_opts=symbolize=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SYMBOLIZE-ON

// RUN: %clangxx_asan -O0 %s -o %t -DUSER_FUNCTION_ON
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SYMBOLIZE-ON
// RUN: %env_asan_opts=symbolize=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SYMBOLIZE-OFF
// RUN: %env_asan_opts=symbolize=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SYMBOLIZE-ON
#if USER_FUNCTION_OFF

extern "C" __declspec(dllexport) extern const char *__asan_default_options() {
  return "symbolize=0";
}

#endif

#if USER_FUNCTION_ON

extern "C" __declspec(dllexport) extern const char *__asan_default_options() {
  return "symbolize=1";
}

#endif

#include <cstdio>
#include <cstdlib>

volatile static int heapBufferOverflowValue = 10;
int main() {
  int *array = new int[10];
  heapBufferOverflowValue = array[10]; // CHECK-SYMBOLIZE-ON: symbolize.cpp:36
  return 0; // CHECK-SYMBOLIZE-OFF: symbolize.cpp.tmp+0x
}
