// RUN: %clang -x c   -fsanitize=alignment -fopenmp-simd -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"
// RUN: %clang -x c   -fsanitize=alignment -fopenmp-simd -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"
// RUN: %clang -x c   -fsanitize=alignment -fopenmp-simd -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"
// RUN: %clang -x c   -fsanitize=alignment -fopenmp-simd -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"

// RUN: %clangxx -x c++ -fsanitize=alignment -fopenmp-simd -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"
// RUN: %clangxx -x c++ -fsanitize=alignment -fopenmp-simd -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"
// RUN: %clangxx -x c++ -fsanitize=alignment -fopenmp-simd -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"
// RUN: %clangxx -x c++ -fsanitize=alignment -fopenmp-simd -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="runtime error:"

#include <stdlib.h>

int main(int argc, char* argv[]) {
  char *ptr = (char *)malloc(2);
  char *data = ptr + 1;

  data[0] = 0;

#pragma omp for simd aligned(data : 0x8000)
  for(int x = 0; x < 1; x++)
    data[x] = data[x];
  // CHECK: {{.*}}align-assume-{{.*}}.cpp:[[@LINE-3]]:30: runtime error: assumption of 32768 byte alignment for pointer of type 'char *' failed
  // CHECK: 0x{{.*}}: note: address is {{.*}} aligned, misalignment offset is {{.*}} byte

  free(ptr);

  return 0;
}
