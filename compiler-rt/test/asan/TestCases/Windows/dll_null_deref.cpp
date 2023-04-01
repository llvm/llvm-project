// UNSUPPORTED: target={{.*-windows-gnu}}

// RUN: %clang_cl_asan -Od %p/dll_host.cpp -Fe%t
// RUN: %clang_cl_asan -LD -Od %s -Fe%t.dll
// RUN: not %run %t %t.dll 2>&1 | FileCheck %s

__attribute__((noinline))
static void NullDeref(int *ptr) {
  // CHECK: ERROR: AddressSanitizer: access-violation on unknown address
  // CHECK:   {{0x0*000.. .*pc 0x.*}}
  ptr[10]++;  // BOOM
}

extern "C" __declspec(dllexport)
int test_function() {
  NullDeref((int*)0);
  // CHECK: {{    #1 0x.* in test_function .*\dll_null_deref.cpp:}}[[@LINE-1]]
  // CHECK: AddressSanitizer can not provide additional info.
  return 0;
}
