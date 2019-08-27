// Test that asan_symbolize does not hang when provided with an non-existing
// path.
// RUN: echo '#0 0xabcdabcd (%t/bad/path+0x1234)' | %asan_symbolize | FileCheck %s -check-prefix CHECK-BAD-FILE
// CHECK-BAD-FILE: #0 0xabcdabcd in ?? ??:0
// CHECK-BAD-FILE-EMPTY:

// Also test that asan_symbolize doesn't assert on an invalid address with a valid file:
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: echo '#0 0xabcdabcd (%t+0x0)' | %asan_symbolize | FileCheck %s -check-prefix CHECK-BAD-ADDR
// CHECK-BAD-ADDR: #0 0xabcdabcd in ??
// CHECK-BAD-ADDR-EMPTY:

int main() {
  return 0;
}
