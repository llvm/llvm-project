// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -emit-pch -o %t/header.pch %t/header.h -x c++
// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -std=c++20 -include-pch %t/header.pch -verify %t/main.cpp

//--- header.h
int foo(int *p) {
  return p[5];  // This will be warned
}

#pragma clang unsafe_buffer_usage begin // The opt-out region spans over two files of one TU
#include "header-2.h"


//--- header-2.h
int bar(int *p) {
  return p[5];  // suppressed by the cross-file opt-out region
}
#pragma clang unsafe_buffer_usage end

//--- main.cpp
// expected-warning@header.h:2 {{unsafe buffer access}}
// expected-note@header.h:2 {{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
