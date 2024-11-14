// The original example from https://github.com/llvm/llvm-project/issues/90501

// Test without PCH
// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -std=c++20 -include %s -verify %s
// Test with PCH
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -emit-pch -o %t %s
// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -std=c++20 -include-pch %t -verify %s

#ifndef A_H
#define A_H

int a(int *s) {
  s[2];  // <- expected warning here
#pragma clang unsafe_buffer_usage begin
  return s[1];
#pragma clang unsafe_buffer_usage end
}

#else
// expected-warning@-7{{unsafe buffer access}}
// expected-note@-8{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
int main() {
  int s[] = {1, 2, 3};
  return a(s);
}

#endif
