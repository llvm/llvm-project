// RUN: %clang_cl_asan -Od %p/dll_host.cpp -Fe%t
// RUN: %clang_cl_asan -LD -Od %s -Fe%t.dll
// RUN: not %run %t %t.dll 2>&1 | FileCheck %s

struct C {
  int x;
  ~C() {}
};

extern "C" __declspec(dllexport)
int test_function() {
  C *buffer = new C[42];
  buffer[-(1 + sizeof(void*) / 4)].x = 42;
// CHECK: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]] thread T0
// CHECK-NEXT: test_function {{.*}}dll_operator_array_new_with_dtor_left_oob.cpp:[[@LINE-3]]
// CHECK-NEXT: main {{.*}}dll_host.cpp
//
// FIXME: Currently it says "4 bytes ... left of 172-byte region",
//        should be "8 bytes ... left of 168-byte region", see
//        https://code.google.com/p/address-sanitizer/issues/detail?id=314
// CHECK: [[ADDR]] is located {{.*}} bytes to the left of {{(172|176)}}-byte region
// FIXME: Should get rid of the malloc/free frames called from the inside of
// operator new/delete in DLLs when using -MT CRT.
// FIXME: The operator new frame should have [].
// CHECK-LABEL: allocated by thread T0 here:
// CHECK:        operator new
// CHECK-NEXT:   test_function {{.*}}dll_operator_array_new_with_dtor_left_oob.cpp:[[@LINE-16]]
// CHECK-NEXT:   main {{.*}}dll_host.cpp
// CHECK-LABEL: SUMMARY
  delete [] buffer;
  return 0;
}
