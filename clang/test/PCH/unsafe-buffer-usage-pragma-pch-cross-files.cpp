// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -emit-pch -o %t/header.pch %t/header.h -x c++
// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -std=c++20 -include-pch %t/header.pch -verify %t/main.cpp

//--- header.h
int foo(int *p) {
  return p[5];  // This will be warned
}

#pragma clang unsafe_buffer_usage begin
#include "header-2.h"
#pragma clang unsafe_buffer_usage end

//--- header-2.h
// Included by the PCH in the traditional way.  The include directive
// in the PCH is enclosed in an opt-out region, so unsafe operations
// here is suppressed.

int bar(int *p) {
  return p[5];
}


//--- main.cpp
// expected-warning@header.h:2 {{unsafe buffer access}}
// expected-note@header.h:2 {{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
