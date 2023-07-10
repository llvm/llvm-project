// REQUIRES: !system-windows

// RUN: if pgrep -f "cc1modbuildd scan"; then pkill -f "cc1modbuildd scan"; fi
// RUN: rm -rf scan
// RUN: split-file %s %t

//--- main.c
#include "header.h"
int main() {return 0;}

//--- header.h

// RUN: %clang -fmodule-build-daemon=scan %t/main.c
// RUN: cat scan/mbd.out | FileCheck %s -DPREFIX=%t

// CHECK: main.c
// CHECK: header.h

// RUN: if pgrep -f "cc1modbuildd scan"; then pkill -f "cc1modbuildd scan"; fi
// RUN: rm -rf scan
