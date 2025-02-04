// Test that the specified output has profiling data.
// RUN: %clang_profgen -fcoverage-mapping -o %t %s
// RUN: %run %t %t.file.profraw
// RUN: test -f %t.file.profraw
// RUN: llvm-profdata merge -o %t.file.profdata %t.file.profraw
// RUN: llvm-cov show -instr-profile %t.file.profdata %t | FileCheck %s --match-full-lines
// RUN: rm %t.file.profraw %t.file.profdata
#include <stdio.h>

extern void __llvm_profile_set_file_object(FILE *, int);

int main(int argc, const char *argv[]) {
  if (argc < 2)
    return 1;

  FILE *F = fopen(argv[1], "w+b");
  __llvm_profile_set_file_object(F, 0);
  return 0;
}
// XFAIL: target={{.*}}-aix{{.*}}
// CHECK:    8|       |#include <stdio.h>
// CHECK:    9|       |
// CHECK:   10|       |extern void __llvm_profile_set_file_object(FILE *, int);
// CHECK:   11|       |
// CHECK:   12|      1|int main(int argc, const char *argv[]) {
// CHECK:   13|      1|  if (argc < 2)
// CHECK:   14|      0|    return 1;
// CHECK:   15|       |
// CHECK:   16|      1|  FILE *F = fopen(argv[1], "w+b");
// CHECK:   17|      1|  __llvm_profile_set_file_object(F, 0);
// CHECK:   18|      1|  return 0;
// CHECK:   19|      1|}
