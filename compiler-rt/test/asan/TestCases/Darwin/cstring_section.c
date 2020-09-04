// Test that AddressSanitizer moves constant strings into a separate section.

// RUN: %clang_asan -c -o %t %s
// RUN: llvm-objdump -s %t | FileCheck %s

// Check that "Hello.\n" is in __asan_cstring and not in __cstring.
// CHECK: Contents of section __asan_cstring:
// CHECK: 48656c6c {{.*}} Hello.
// CHECK: Contents of section {{.*}}__const:
// CHECK-NOT: 48656c6c {{.*}} Hello.
// CHECK: Contents of section {{.*}}__cstring:
// CHECK-NOT: 48656c6c {{.*}} Hello.

int main(int argc, char *argv[]) {
  argv[0] = "Hello.\n";
  return 0;
}
