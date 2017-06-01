// Check to make sure argv[0] is contained within the (randomised) XRay log file
// name.

// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: %run %t > xray.log.file.name 2>&1
// RUN: ls | FileCheck xray.log.file.name
// RUN: rm xray-log.* xray.log.file.name

#include <cstdio>
#include <libgen.h>

[[clang::xray_always_instrument]] int main(int argc, char *argv[]) {
  printf("// CHECK: xray-log.%s.{{.*}}\n", basename(argv[0]));
}
