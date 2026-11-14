// RUN: %clang_dasan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %clang_dasan -shared-libsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// The runtime exports the HSA wrappers to every program. Without HSA loaded
// they must report failure rather than kill the process.

#include <dlfcn.h>
#include <stdio.h>

int main(void) {
  int (*Init)(void) = (int (*)(void))dlsym(RTLD_DEFAULT, "hsa_init");
  int (*ShutDown)(void) = (int (*)(void))dlsym(RTLD_DEFAULT, "hsa_shut_down");
  if (!Init || !ShutDown)
    return 1;
  fprintf(stderr, "init: %s\n", Init() ? "failed" : "succeeded");
  fprintf(stderr, "shut_down: %s\n", ShutDown() ? "failed" : "succeeded");
  return 0;
}

// CHECK: init: failed
// CHECK: shut_down: failed
