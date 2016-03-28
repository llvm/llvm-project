// RUN: %clang_profgen -o %t -O3 %s
// RUN: LLVM_PROFILE_VERBOSE_ERRORS=1 %run %t 1 2>&1 | FileCheck %s

// override the version variable with a bogus version:
unsigned long long __llvm_profile_raw_version = 10000;
int main(int argc, const char *argv[]) {
  if (argc < 2)
    return 1;
  return 0;
}
// CHECK: LLVM Profile: runtime and instrumentation version mismatch
