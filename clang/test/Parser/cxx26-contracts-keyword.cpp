// RUN: %clang_cc1 -std=c++2c -fcontracts -DCXX26 -fsyntax-only -verify=enabled,expected %s
// RUN: %clang_cc1 -std=c++2c -fno-contracts -DCXX26 -fsyntax-only -verify=disabled,expected %s
// RUN: %clang_cc1 -std=c++23 -fcontracts -fsyntax-only -verify=precxx26,expected %s

#ifdef __cplusplus
#ifdef CXX26
void test() {
  int contract_assert; // expected-error {{expected unqualified-id}}
}

void assertion() {
  contract_assert(true); // disabled-error {{contracts support is disabled; pass '-fcontracts' to enable it}}
}

int specifier(int value) pre(value > 0);
// disabled-error@-1 {{contracts support is disabled; pass '-fcontracts' to enable it}}
#else
int old_standard(int value) pre(value > 0);
// precxx26-error@-1 {{contracts are a C++26 feature}}
#endif
#endif
