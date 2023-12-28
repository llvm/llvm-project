// RUN: %clang_pgogen -o %t %s
// RUN: not %t
// RUN: %clang -o %t %s
// RUN: %t

__attribute__((weak)) void __llvm_profile_reset_counters(void);

__attribute__((noinline)) int bar() { return 4; }
int foo() {
  if (__llvm_profile_reset_counters) {
    __llvm_profile_reset_counters();
    return 0;
  }
  return bar();
}

int main() { return foo() - 4; }
