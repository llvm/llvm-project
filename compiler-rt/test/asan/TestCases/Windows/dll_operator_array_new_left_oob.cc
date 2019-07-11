// RUN: %clang_cl_asan -Od %p/dll_host.cc -Fe%t
// RUN: %clang_cl_asan -LD -Od %s -Fe%t.dll
// RUN: not %run %t %t.dll 2>&1 | FileCheck %s

extern "C" __declspec(dllexport)
int test_function() {
  char *buffer = new char[42];
  buffer[-1] = 42;
// CHECK: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] thread T0
// CHECK-NEXT: test_function {{.*}}dll_operator_array_new_left_oob.cc:[[@LINE-3]]
// CHECK-NEXT: main {{.*}}dll_host.cc
//
// CHECK: [[ADDR]] is located 1 bytes to the left of 42-byte region
// CHECK-LABEL: allocated by thread T0 here:
// FIXME: Should get rid of the malloc/free frames called from the inside of
// operator new/delete in DLLs when using -MT CRT.
// FIXME: The 'operator new' frame should have [].
// CHECK:        operator new
// CHECK-NEXT:   test_function {{.*}}dll_operator_array_new_left_oob.cc:[[@LINE-13]]
// CHECK-NEXT:   main {{.*}}dll_host.cc
// CHECK-LABEL: SUMMARY
  delete [] buffer;
  return 0;
}
