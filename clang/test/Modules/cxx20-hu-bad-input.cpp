// Test generation and import of simple C++20 Header Units.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: not %clang_cc1 -std=c++20 -emit-header-unit \
// RUN:  -xc++-header-unit-header hu-01.hh \
// RUN:  -xc++-header-unit-header hu-02.hh \
// RUN:  -o hu-01.pcm -verify  2>&1 | FileCheck %s

// CHECK: (frontend): multiple inputs are not valid for header units (first extra 'hu-02.hh')

//--- hu-01.hh
int foo(int);

//--- hu-02.hh
int bar(int);
