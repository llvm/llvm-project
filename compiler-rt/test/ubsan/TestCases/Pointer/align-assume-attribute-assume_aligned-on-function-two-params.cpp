// RUN: %clang   -x c   -fsanitize=alignment -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"
// RUN: %clang   -x c   -fsanitize=alignment -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"
// RUN: %clang   -x c   -fsanitize=alignment -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"
// RUN: %clang   -x c   -fsanitize=alignment -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"

// RUN: %clangxx -x c++ -fsanitize=alignment -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"
// RUN: %clangxx -x c++ -fsanitize=alignment -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"
// RUN: %clangxx -x c++ -fsanitize=alignment -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"
// RUN: %clangxx -x c++ -fsanitize=alignment -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"

#include <stdlib.h>

char *__attribute__((assume_aligned(0x8000, 1))) passthrough(char *x) {
  return x;
}

int main(int argc, char* argv[]) {
  char *ptr = (char *)malloc(3);

  passthrough(ptr + 2);
  // CHECK: {{.*}}align-assume-{{.*}}.cpp:[[@LINE-1]]:3: runtime error: assumption of 32768 byte alignment (with offset of 1 byte) for pointer of type 'char *' failed
  // CHECK: {{.*}}align-assume-{{.*}}.cpp:[[@LINE-9]]:22: note: alignment assumption was specified here
  // CHECK: 0x{{.*}}: note: offset address is {{.*}} aligned, misalignment offset is {{.*}} byte

  free(ptr);

  return 0;
}
