// Test PCHs:
//   MAIN - includes textual_1.h
//        \ loads    pch_1.h - includes textual_2.h
//                           \ loads    pch_2.h

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -emit-pch -o %t/pch_2.h.pch %t/pch_2.h -x c++ -Wunsafe-buffer-usage -verify
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -include-pch %t/pch_2.h.pch -emit-pch -o %t/pch_1.h.pch %t/pch_1.h -x c++ -Wunsafe-buffer-usage -verify
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -include-pch %t/pch_1.h.pch -verify %t/main.cpp -Wunsafe-buffer-usage

// With '-fsafe-buffer-usage-suggestions':

// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -emit-pch -o %t/pch_2.h.pch %t/pch_2.h -x c++ -Wunsafe-buffer-usage -verify=with-fixit -fsafe-buffer-usage-suggestions
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -include-pch %t/pch_2.h.pch -emit-pch -o %t/pch_1.h.pch %t/pch_1.h -x c++ -Wunsafe-buffer-usage -verify=with-fixit -fsafe-buffer-usage-suggestions
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -include-pch %t/pch_1.h.pch %t/main.cpp -Wunsafe-buffer-usage -verify=with-fixit -fsafe-buffer-usage-suggestions


//--- textual_1.h
int a(int *s) {
  s[2];  // <- expected warning here
#pragma clang unsafe_buffer_usage begin
  return s[1];
#pragma clang unsafe_buffer_usage end

}

//--- textual_2.h
int b(int *s) {
  s[2];  // <- expected warning here
#pragma clang unsafe_buffer_usage begin
  return s[1];
#pragma clang unsafe_buffer_usage end
}

//--- pch_1.h
#include "textual_2.h" // with-fixit-no-diagnostics
// expected-warning@textual_2.h:2{{unsafe buffer access}} \
   expected-note@textual_2.h:2{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
int c(int *s) {
  s[2];  // expected-warning{{unsafe buffer access}} \
	    expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
#pragma clang unsafe_buffer_usage begin
  return s[1];
#pragma clang unsafe_buffer_usage end
}

//--- pch_2.h
int d(int *s) { // with-fixit-no-diagnostics
  s[2];  // expected-warning{{unsafe buffer access}} \
	    expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
#pragma clang unsafe_buffer_usage begin
  return s[1];
#pragma clang unsafe_buffer_usage end
}


//--- main.cpp
#include "textual_1.h"
// expected-warning@textual_1.h:2{{unsafe buffer access}} \
   expected-note@textual_1.h:2{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}

// with-fixit-warning@textual_1.h:1 {{'s' is an unsafe pointer used for buffer access}}
// with-fixit-note@textual_1.h:2 {{used in buffer access here}}
// with-fixit-warning@textual_2.h:1 {{'s' is an unsafe pointer used for buffer access}}
// with-fixit-note@textual_2.h:2 {{used in buffer access here}}
// with-fixit-warning@pch_1.h:4 {{'s' is an unsafe pointer used for buffer access}}
// with-fixit-note@pch_1.h:5 {{used in buffer access here}}
// with-fixit-warning@pch_2.h:1 {{'s' is an unsafe pointer used for buffer access}}
// with-fixit-note@pch_2.h:2 {{used in buffer access here}}
int main() {
  int s[] = {1, 2, 3};
  return a(s) + b(s) + c(s) + d(s);
}
