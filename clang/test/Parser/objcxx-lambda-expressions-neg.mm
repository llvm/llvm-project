// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify -std=c++11 %s

int main() {
  []{};
#if __cplusplus <= 199711L
  // expected-error@-2 {{expected expression}}
#else
  // expected-no-diagnostics
#endif

}
