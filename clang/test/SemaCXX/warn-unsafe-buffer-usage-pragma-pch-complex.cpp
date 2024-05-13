// A more complex example than warn-unsafe-buffer-usage-pragma-pch.cpp:
//   MAIN - includes INC_H_1 
//        \ loads    PCH_H_1 - includes INC_H_2
//                           \ loads    PCH_H_2   

// Test with PCH
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -emit-pch -o %t-1 -DPCH_H_2 %s
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -include-pch %t-1 -emit-pch -o %t-2 -DPCH_H_1 %s
// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -std=c++20 -include-pch %t-2 -DMAIN -verify %s

#define UNSAFE_BEGIN _Pragma("clang unsafe_buffer_usage begin")
#define UNSAFE_END   _Pragma("clang unsafe_buffer_usage end")


#ifdef INC_H_1
#undef INC_H_1
int a(int *s) {
  s[2];  // <- expected warning here
#pragma clang unsafe_buffer_usage begin
  return s[1];
#pragma clang unsafe_buffer_usage end
}
#endif

#ifdef INC_H_2
#undef INC_H_2
int b(int *s) {
  s[2];  // <- expected warning here  
#pragma clang unsafe_buffer_usage begin
  return s[1];
#pragma clang unsafe_buffer_usage end
}
#endif

#ifdef PCH_H_1
#undef PCH_H_1
#define INC_H_2
#include "warn-unsafe-buffer-usage-pragma-pch-complex.cpp"

int c(int *s) {
  s[2];  // <- expected warning here  
UNSAFE_BEGIN
  return s[1];
UNSAFE_END
}
#endif

#ifdef PCH_H_2
#undef PCH_H_2
int d(int *s) {
  s[2];  // <- expected warning here  
#pragma clang unsafe_buffer_usage begin
  return s[1];
#pragma clang unsafe_buffer_usage end
}
#endif

#ifdef MAIN
#undef MAIN
#define INC_H_1
#include "warn-unsafe-buffer-usage-pragma-pch-complex.cpp"

// expected-warning@-45{{unsafe buffer access}} expected-note@-45{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
// expected-warning@-36{{unsafe buffer access}} expected-note@-36{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
// expected-warning@-24{{unsafe buffer access}} expected-note@-24{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
// expected-warning@-15{{unsafe buffer access}} expected-note@-15{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
int main() {
  int s[] = {1, 2, 3};
  return a(s) + b(s) + c(s) + d(s);
}
#undef MAIN
#endif
