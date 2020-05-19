// Check to make sure argv[0] is contained within the (randomised) XRay log file
// name.

// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: XRAY_OPTIONS="patch_premain=true xray_mode=xray-basic" %run %t > xray.log.file.name 2>&1
// RUN: ls | FileCheck xray.log.file.name
// RUN: rm xray-log.argv0-log-file-name.* xray.log.file.name

// UNSUPPORTED: target-is-mips64,target-is-mips64el

#include <cstdio>
#include <libgen.h>

[[clang::xray_always_instrument]] int main(int argc, char *argv[]) {
  printf("// CHECK: xray-log.%s.{{.*}}\n", basename(argv[0]));
}
