// Test that unprofiled files are recognized. Here, we have two functions in the
// profile, main() and function_in_header, but we use the profile on a file that
// has the profile-less some_unprofiled_function so that the only profiled code
// in #included in a header.

// FIXME: It would be nice to use -verify here instead of FileCheck, but -verify
// doesn't play well with warnings that have no line number.

// RUN: llvm-profdata merge %S/Inputs/c-unprofiled.proftext -o %t.profdata
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-unprofiled.c -I %S/Inputs/ %s -o /dev/null -emit-llvm -fprofile-instrument-use-path=%t.profdata -Wprofile-instr-unprofiled 2>&1 | FileCheck %s

// CHECK: warning: no profile data available for file "c-unprofiled.c"

#include "profiled_header.h"

#ifdef GENERATE_OUTDATED_DATA
int main(int argc, const char *argv[]) {
  function_in_header(0);
  return 0;
}
#else
void some_unprofiled_function(int i) {
  if (i)
    function_in_header(i);
}
#endif
