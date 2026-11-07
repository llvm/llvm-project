// Check that a header textually entered by several header units reuses the
// source location entries of the header unit that wrote it first.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -xc++-user-header -I%t \
// RUN:   -emit-header-unit %t/hu1.h -o %t/hu1.pcm
// RUN: %clang_cc1 -std=c++20 -xc++-user-header -I%t -fmodule-file=%t/hu1.pcm \
// RUN:   -emit-header-unit %t/hu2.h -o %t/hu2.pcm

// RUN: llvm-bcanalyzer -dump %t/hu1.pcm | FileCheck %s --check-prefix=WROTE
// RUN: llvm-bcanalyzer -dump %t/hu2.pcm | FileCheck %s --check-prefix=REUSED

// The input file record keeps the index of the first source location entry
// written for the file. It is zero when the header unit reuses a copy that a
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

//--- hu1.h
#include "shared.h"
int hu1_fn(void);

//--- hu2.h
#include "shared.h"
int hu2_fn(void);
