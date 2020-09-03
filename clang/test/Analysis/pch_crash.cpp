// RUN: %clang_cc1 -triple x86_64-apple-macosx10.15.0 -emit-pch -o %t %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-macosx10.15.0 -include-pch %t \
// RUN:   -analyzer-checker=core,apiModeling -verify %s
//
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_analyze_cc1 -include-pch %t \
// RUN:   -analyzer-checker=core,apiModeling -verify %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER
// Pre-compiled header

int foo();

// Literal data for this macro value will be null
#define EOF -1

#else
// Source file

int test() {
  // we need a function call here to initiate erroneous routine
  return foo(); // no-crash
}

#endif
