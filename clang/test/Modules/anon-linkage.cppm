// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -fsyntax-only -verify

//--- foo.h
typedef struct {
  int c;
  union {
    int n;
    char c[4];
  } v;
} mbstate;

//--- M.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module M;
export using ::mbstate;
