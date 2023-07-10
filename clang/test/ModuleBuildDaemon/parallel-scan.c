// Confirm module build daemon can handle two translation units simultaneously

// REQUIRES: !system-windows

// RUN: if pgrep -f "cc1modbuildd parallel-scan"; then pkill -f "cc1modbuildd parallel-scan"; fi
// : rm -rf parallel-scan
// RUN: split-file %s %t

//--- main.c
#include "app.h"
int main() {return 0;}

//--- app.c
#include "util.h"

//--- app.h

//--- util.h

// RUN: %clang -fmodule-build-daemon=parallel-scan %t/main.c %t/app.c
// RUN: pwd && ls 
// RUN: cat parallel-scan/mbd.out
// RUN: cat parallel-scan/mbd.out | FileCheck %s -DPREFIX=%t

// CHECK: main.c
// CHECK: app.h
// CHECK: app.c
// CHECK: util.h

// RUN: if pgrep -f "cc1modbuildd parallel-scan"; then pkill -f "cc1modbuildd parallel-scan"; fi
// : rm -rf parallel-scan
