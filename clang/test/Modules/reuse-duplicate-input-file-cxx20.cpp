// Check that a header textually entered by several C++20 named modules reuses
// the source location entries of the module that wrote it first.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -I%t -emit-module-interface %t/m1.cppm -o %t/m1.pcm
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=m1=%t/m1.pcm \
// RUN:   -emit-module-interface %t/m2.cppm -o %t/m2.pcm

// RUN: llvm-bcanalyzer -dump %t/m1.pcm | FileCheck %s --check-prefix=WROTE
// RUN: llvm-bcanalyzer -dump %t/m2.pcm | FileCheck %s --check-prefix=REUSED

// Reduced BMIs record input files the same way.
// RUN: %clang_cc1 -std=c++20 -I%t -emit-reduced-module-interface %t/m1.cppm \
// RUN:   -o %t/m1-reduced.pcm
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=m1=%t/m1-reduced.pcm \
// RUN:   -emit-reduced-module-interface %t/m2.cppm -o %t/m2-reduced.pcm

// RUN: llvm-bcanalyzer -dump %t/m1-reduced.pcm | FileCheck %s --check-prefix=WROTE
// RUN: llvm-bcanalyzer -dump %t/m2-reduced.pcm | FileCheck %s --check-prefix=REUSED

// The input file record keeps the index of the first source location entry
// written for the file. It is zero when the module reuses a copy that a
// loaded module already has, and nonzero when it wrote its own entries.

// WROTE: <INPUT_FILE {{.*}}op7={{[1-9][0-9]*}} {{.*}}shared.h
// REUSED: <INPUT_FILE {{.*}}op7=0 op8=0{{.*}}shared.h

//--- shared.h
#ifndef SHARED_H
#define SHARED_H
struct Shared {
  int a;
};
int shared_fn(void);
#endif

//--- m1.cppm
module;
#include "shared.h"
export module m1;
export int m1_fn();

//--- m2.cppm
module;
#include "shared.h"
export module m2;
import m1;
export int m2_fn();
