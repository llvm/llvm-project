// RUN: %clang_scs -D INCLUDE_RUNTIME %s -o %t
// RUN: %run %t

// RUN: %clang_scs %s -o %t
// RUN: not --crash %run %t

// Basic smoke test for the runtime

#include "libc_support.h"

#ifdef INCLUDE_RUNTIME
#include "minimal_runtime.h"
#else
#define scs_main main
#endif

int scs_main(void) {
  scs_fputs_stdout("In main.\n");
  return 0;
}
