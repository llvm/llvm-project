// XFAIL: target={{.*haiku.*}}
// RUN: %clang_profgen -o %t -O3 %s
// RUN: env LLVM_PROFILE_FILE=%t/  %run %t 1 2>&1 | FileCheck %s

int main(int argc, const char *argv[]) {
  if (argc < 2)
    return 1;
  return 0;
}
// CHECK: LLVM Profile Error: Failed to write file 
