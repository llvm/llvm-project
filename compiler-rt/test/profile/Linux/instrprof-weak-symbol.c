// Test the linker feature that treats undefined weak symbols as null values.

// RUN: %clang_pgogen -o %t %s
// RUN: not %t
// RUN: %clang -o %t %s
// RUN: %t

__attribute__((weak)) void __llvm_profile_reset_counters(void);

int main() {
  if (__llvm_profile_reset_counters) {
    __llvm_profile_reset_counters();
    return 1;
  }
  return 0;
}
