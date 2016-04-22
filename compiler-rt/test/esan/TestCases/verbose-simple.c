// RUN: %clang_esan_frag -O0 %s -o %t 2>&1
// RUN: %env_esan_opts=verbosity=1 %run %t 2>&1 | FileCheck %s

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  // CHECK:      in esan::initializeLibrary
  // CHECK-NEXT: in esan::finalizeLibrary
  // CHECK-NEXT: {{.*}}EfficiencySanitizer is not finished: nothing yet to report
  return 0;
}
