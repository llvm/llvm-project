// Test that the specified output merges the profiling data.
// Run the program twice so that the counters accumulate.
// RUN: %clang_profgen -fcoverage-mapping -o %t %s
// RUN: rm -f %t.merging.profraw %t.merging.profdata
// RUN: %run %t %t.merging.profraw
// RUN: %run %t %t.merging.profraw
// RUN: test -f %t.merging.profraw
// RUN: llvm-profdata merge -o %t.merging.profdata %t.merging.profraw
// RUN: llvm-cov show -instr-profile %t.merging.profdata %t | FileCheck %s --match-full-lines
#include <stdio.h>

extern void __llvm_profile_set_file_object(FILE *, int);

int main(int argc, const char *argv[]) {
  if (argc < 2)
    return 1;

  FILE *F = fopen(argv[1], "r+b");
  if (!F) {
    // File might not exist, try opening with truncation
    F = fopen(argv[1], "w+b");
  }
  __llvm_profile_set_file_object(F, 1);

  return 0;
}
// CHECK:   10|       |#include <stdio.h>
// CHECK:   11|       |
// CHECK:   12|       |extern void __llvm_profile_set_file_object(FILE *, int);
// CHECK:   13|       |
// CHECK:   14|      2|int main(int argc, const char *argv[]) {
// CHECK:   15|      2|  if (argc < 2)
// CHECK:   16|      0|    return 1;
// CHECK:   17|       |
// CHECK:   18|      2|  FILE *F = fopen(argv[1], "r+b");
// CHECK:   19|      2|  if (!F) {
// CHECK:   20|       |    // File might not exist, try opening with truncation
// CHECK:   21|      1|    F = fopen(argv[1], "w+b");
// CHECK:   22|      1|  }
// CHECK:   23|      2|  __llvm_profile_set_file_object(F, 1);
// CHECK:   24|       |
// CHECK:   25|      2|  return 0;
// CHECK:   26|      2|}
