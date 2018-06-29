// RUN: %clang_scs %s -o %t
// RUN: %run %t

// Basic smoke test for the runtime

#include "libc_support.h"
#include "minimal_runtime.h"

int scs_main(void) {
  scs_fputs_stdout("In main.\n");
  return 0;
}
