// RUN: %clangxx %s -o %t && %run not %t 1 2>&1 | FileCheck %s
// UNSUPPORTED: lsan,ubsan,android

#include <dlfcn.h>
#include <stdio.h>
#include <string>

int main (int argc, char *argv[]) {
  // CHECK: You are trying to dlopen a <arbitrary path> shared library with RTLD_DEEPBIND flag
  void *lib = dlopen("<arbitrary path>", RTLD_NOW | RTLD_DEEPBIND);
  return 0;
}
