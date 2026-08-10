// Test that AddressSanitizer moves constant strings into a separate section.

// RUN: %clang_asan -c -o %t %s
// RUN: llvm-objdump -s %t | FileCheck %s --implicit-check-not="Hello."

// Check that "Hello.\n" is in __asan_cstring and not in __cstring.
// CHECK: Contents of section {{.*}}__asan_cstring:
// CHECK-NEXT: 48656c6c {{.*}} Hello.

int main(int argc, char *argv[]) {
  argv[0] = "Hello.\n";
  return 0;
}
