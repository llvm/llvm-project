// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -emit-pch -o %t/header.pch %t/header.h -x c++ -verify -Wunsafe-buffer-usage
// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -std=c++20 -include-pch %t/header.pch -verify=with-fixit %t/main.cpp -fsafe-buffer-usage-suggestions
// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -std=c++20 -include-pch %t/header.pch %t/main.cpp -verify

//--- header.h
int foo(int *p) {
  return p[5];  // expected-warning{{unsafe buffer access}}\
		   expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
}

#pragma clang unsafe_buffer_usage begin // The opt-out region spans over two prefix files of one TU
#include "header-2.h"


//--- header-2.h
int bar(int *p) {
  return p[5];  // suppressed by the cross-file opt-out region
}
#pragma clang unsafe_buffer_usage end

//--- main.cpp
// with-fixit-warning@header.h:1 {{'p' is an unsafe pointer used for buffer access}}
// with-fixit-note@header.h:1 {{change type of 'p' to 'std::span' to preserve bounds information}}
// with-fixit-note@header.h:2 {{used in buffer access here}}
// expected-no-diagnostics
