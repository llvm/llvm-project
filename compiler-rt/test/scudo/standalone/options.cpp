// RUN: %clangxx_scudo %s -o %t
// RUN:                                         not --crash %run %t 2>&1 | FileCheck %s
// RUN: %env_scudo_opts=dealloc_type_mismatch=0             %run %t 2>&1
// RUN: %env_scudo_opts=dealloc_type_mismatch=1 not --crash %run %t 2>&1 | FileCheck %s

// Tests that the options can be passed using getScudoDefaultOptions, and that
// the environment ones take precedence over them.

#include <assert.h>
#include <malloc.h>
#include <stdlib.h>

extern "C" const char *__scudo_default_options() {
  return "dealloc_type_mismatch=1"; // Defaults to false in flags.inc.
}

int main(int argc, char **argv) {
  int *p = (int *)malloc(16);
  assert(p);
  delete p;
  return 0;
}

// CHECK: ERROR: allocation type mismatch when deallocating address
