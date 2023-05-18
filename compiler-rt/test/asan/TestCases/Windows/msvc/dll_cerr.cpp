// RUN: %clang_cl_asan -Od %p/../dll_host.cpp -Fe%t
// RUN: %clang_cl_asan -LD -Od %s -Fe%t.dll
// RUN: %run %t %t.dll 2>&1 | FileCheck %s

// Test that it works correctly even with ICF enabled.
// RUN: %clang_cl_asan -LD -Od %s -Fe%t.dll -link /OPT:REF /OPT:ICF
// RUN: %run %t %t.dll 2>&1 | FileCheck %s

#include <iostream>

extern "C" __declspec(dllexport)
int test_function() {
  // Just make sure we can use cout.
  std::cout << "All ok\n";
// CHECK: All ok

  // This line forces a declaration of some global basic_ostream internal object that
  // calls memcpy() in its constructor.  This doesn't work if __asan_init is not
  // called early enough.
  std::cout << 42;
// CHECK: 42
  return 0;
}
