// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

int main() {
  char *buffer = new char[42];
  buffer[-1] = 42;
  // CHECK: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
  // CHECK: WRITE of size 1 at [[ADDR]] thread T0
  // CHECK-NEXT: {{#0 .* main .*operator_array_new_left_oob.cpp}}:[[@LINE-3]]
  //
  // CHECK: [[ADDR]] is located 1 bytes before 42-byte region
  // CHECK-LABEL: allocated by thread T0 here:
  // CHECK-NEXT: {{#0 .* operator new}}[]
  // CHECK-NEXT: {{#1 .* main .*operator_array_new_left_oob.cpp}}:[[@LINE-9]]
  delete [] buffer;
}
