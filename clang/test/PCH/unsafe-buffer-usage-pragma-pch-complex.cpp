// Test PCHs:
//   MAIN - includes textual_1.h
//        \ loads    pch_1.h - includes textual_2.h
//                           \ loads    pch_2.h

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -emit-pch -o %t/pch_2.h.pch %t/pch_2.h -x c++
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -include-pch %t/pch_2.h.pch -emit-pch -o %t/pch_1.h.pch %t/pch_1.h -x c++
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -include-pch %t/pch_1.h.pch -verify %t/main.cpp -Wunsafe-buffer-usage


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
#include "textual_2.h"

int c(int *s) {
  s[2];  // <- expected warning here
#pragma clang unsafe_buffer_usage begin
  return s[1];
#pragma clang unsafe_buffer_usage end
}

//--- pch_2.h
int d(int *s) {
  s[2];  // <- expected warning here
#pragma clang unsafe_buffer_usage begin
  return s[1];
#pragma clang unsafe_buffer_usage end
}


//--- main.cpp
#include "textual_1.h"
// expected-warning@textual_1.h:2{{unsafe buffer access}} \
   expected-note@textual_1.h:2{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
// expected-warning@textual_2.h:2{{unsafe buffer access}} \
   expected-note@textual_2.h:2{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
// expected-warning@pch_1.h:4{{unsafe buffer access}} \
   expected-note@pch_1.h:4{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
// expected-warning@pch_2.h:2{{unsafe buffer access}} \
   expected-note@pch_2.h:2{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
int main() {
  int s[] = {1, 2, 3};
  return a(s) + b(s) + c(s) + d(s);
}
