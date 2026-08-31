// The original example from https://github.com/llvm/llvm-project/issues/90501

// Test without PCH
// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -std=c++20 -include %s -verify %s
// Test with PCH
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -emit-pch -o %t %s -Wunsafe-buffer-usage -verify
// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -std=c++20 -include-pch %t -verify=pch-main %s

// Test with PCH and -fsafe-buffer-usage-suggestions
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -emit-pch -o %t %s -Wunsafe-buffer-usage -verify=fixit \
// RUN:            -fsafe-buffer-usage-suggestions
// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -std=c++20 -include-pch %t -verify=pch-main-fixit %s\
// RUN:            -fsafe-buffer-usage-suggestions

#ifndef A_H
#define A_H

int a(int *s) {
  s[2]; // expected-warning {{unsafe buffer access}}\
           expected-note {{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
        // fixit-no-diagnostics
#pragma clang unsafe_buffer_usage begin
  return s[1];
#pragma clang unsafe_buffer_usage end
}

#else
// pch-main-no-diagnostics
// pch-main-fixit-warning@-11{{'s' is an unsafe pointer used for buffer access}}\
   pch-main-fixit-note@-10{{used in buffer access here}}
int main() {
  int s[] = {1, 2, 3};
  return a(s);
}

#endif
