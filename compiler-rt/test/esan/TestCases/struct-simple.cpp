// RUN: %clang_esan_frag -O0 %s -DPART -c -o %t-part.o 2>&1
// RUN: %clang_esan_frag -O0 %s -DMAIN -c -o %t-main.o 2>&1
// RUN: %clang_esan_frag -O0 %t-part.o %t-main.o -o %t 2>&1
// RUN: %env_esan_opts=verbosity=2 %run %t 2>&1 | FileCheck %s

// We generate two different object files from this file with different
// macros, and then link them together. We do this to test how we handle
// separate compilation with multiple compilation units.

#include <stdio.h>

extern "C" {
  void part();
}

//===-- compilation unit without main function ----------------------------===//

#ifdef PART
void part()
{
}
#endif // PART

//===-- compilation unit with main function -------------------------------===//

#ifdef MAIN
int main(int argc, char **argv) {
  // CHECK:      in esan::initializeLibrary
  // CHECK:      in esan::processCompilationUnitInit
  // CHECK:      in esan::processCompilationUnitInit
  part();
  return 0;
  // CHECK-NEXT: in esan::finalizeLibrary
  // CHECK-NEXT: {{.*}}EfficiencySanitizer is not finished: nothing yet to report
  // CHECK:      in esan::processCompilationUnitExit
  // CHECK:      in esan::processCompilationUnitExit
}
#endif // MAIN
