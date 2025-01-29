// Testing shared library support in basic logging mode.

// RUN: split-file %s %t
// RUN: %clangxx_xray -g -fPIC -fxray-instrument -fxray-shared -shared -std=c++11 %t/testlib.cpp -o %t/testlib.so
// RUN: %clangxx_xray -g -fPIC -fxray-instrument -fxray-shared -std=c++11 %t/main.cpp %t/testlib.so -Wl,-rpath,%t -o %t/main.o

// RUN: XRAY_OPTIONS="patch_premain=false,xray_mode=xray-basic,xray_logfile_base=basic-mode-dso-,verbosity=1" XRAY_BASIC_OPTIONS="func_duration_threshold_us=0" %run %t/main.o 2>&1 | FileCheck %s
// RUN: %llvm_xray account --format=csv --sort=funcid "`ls basic-mode-dso-* | head -1`" | FileCheck --check-prefix=ACCOUNT %s
// RUN: rm basic-mode-dso-*

// REQUIRES: target={{(aarch64|x86_64)-.*}}
// REQUIRES: built-in-llvm-tree

//--- main.cpp

#include "xray/xray_interface.h"

#include <cstdio>
#include <unistd.h>

[[clang::xray_always_instrument]] void instrumented_in_executable() {
  printf("instrumented_in_executable called\n");
  sleep(1);
}

extern void instrumented_in_dso();

int main() {
  // Explicit patching to ensure the DSO has been loaded
  __xray_patch();
  instrumented_in_executable();
  // CHECK: instrumented_in_executable called
  instrumented_in_dso();
  // CHECK-NEXT: instrumented_in_dso called
}

//--- testlib.cpp

#include <cstdio>
#include <unistd.h>

[[clang::xray_always_instrument]] void instrumented_in_dso() {
  printf("instrumented_in_dso called\n");
}

// ACCOUNT: funcid,count,min,median,90%ile,99%ile,max,sum,debug,function
// ACCOUNT-NEXT: 1,1,{{.*}}
// ACCOUNT-NEXT: 16777217,1,{{.*}}
