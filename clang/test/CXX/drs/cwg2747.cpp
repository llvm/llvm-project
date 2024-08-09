// RUN: %clang_cc1 -std=c++98 -pedantic-errors -verify=cxx98 %s -E | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -std=c++11 -pedantic-errors -verify=since-cxx11 %s -E | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -std=c++14 -pedantic-errors -verify=since-cxx11 %s -E | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -std=c++17 -pedantic-errors -verify=since-cxx11 %s -E | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -std=c++20 -pedantic-errors -verify=since-cxx11 %s -E | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -std=c++23 -pedantic-errors -verify=since-cxx11 %s -E | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -verify=since-cxx11 %s -E | FileCheck %s --strict-whitespace

// since-cxx11-no-diagnostics
// cwg2747: yes

// Check that a newline is still added even though there is a
// physical newline at the end of the file (which is spliced)
// CHECK: int x;{{$[[:space:]]^}}int y;int z;{{$[[:space:]]^$}}
// cxx98-error@+4 {{no newline at end of file}}
// cxx98-note@+3 {{last newline deleted by splice here}}
int x;
int y;\
int z;\
