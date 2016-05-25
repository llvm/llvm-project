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
  // CHECK:      in esan::initializeCacheFrag
  // CHECK-NEXT: in esan::processCompilationUnitInit
  // CHECK-NEXT: in esan::processCacheFragCompilationUnitInit
  // CHECK-NEXT: in esan::processCompilationUnitInit
  // CHECK-NEXT: in esan::processCacheFragCompilationUnitInit
  part();
  return 0;
  // CHECK:      in esan::finalizeLibrary
  // CHECK-NEXT: in esan::finalizeCacheFrag
  // CHECK-NEXT: {{.*}}EfficiencySanitizer is not finished: nothing yet to report
  // CHECK-NEXT: in esan::processCompilationUnitExit
  // CHECK-NEXT: in esan::processCacheFragCompilationUnitExit
  // CHECK-NEXT: in esan::processCompilationUnitExit
  // CHECK-NEXT: in esan::processCacheFragCompilationUnitExit
}
#endif // MAIN
