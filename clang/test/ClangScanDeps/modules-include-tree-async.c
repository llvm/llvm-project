// This test checks that we only create single scanning module in asynchronous
// include-tree scans.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: clang-scan-deps -format experimental-include-tree-full -cas-path %t/cas -async-scan-modules -- \
// RUN:   %clang -fmodules -fmodules-cache-path=%t/cache -c %t/tu.c -o %t/tu.o
// RUN: find %t/cache -name '*.pcm' | wc -l | grep 1

//--- tu.c
#include "m.h"
//--- m.h
//--- module.modulemap
module m { header "m.h" }
