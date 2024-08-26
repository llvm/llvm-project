// REQUIRES: darwin

// RUN: %clang_profgen -fcoverage-mapping -fcoverage-mcdc -O3 -o %t.exe %s
// RUN: env LLVM_PROFILE_FILE="%c%t.profraw" %run %t.exe 3 3
// RUN: llvm-profdata show --text --all-functions %t.profraw | FileCheck %s

// CHECK: Num Bitmap Bytes:
// CHECK-NEXT: $1
// CHECK-NEXT: Bitmap Byte Values:
// CHECK-NEXT: 0x4
#include <stdio.h>
#include <stdlib.h>
extern int __llvm_profile_is_continuous_mode_enabled(void);
int main(int argc, char *const argv[]) {
  if (!__llvm_profile_is_continuous_mode_enabled())
    return 1;

  if (argc < 3)
    return 1;

  if ((atoi(argv[1]) > 2) && (atoi(argv[2]) > 2)) {
    printf("Decision Satisfied");
  }

  return 0;
}
