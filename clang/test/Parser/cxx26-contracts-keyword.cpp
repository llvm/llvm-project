// RUN: %clang_cc1 -std=c++2c -fcontracts -DENABLED=1 -fsyntax-only -verify=enabled %s
// RUN: %clang_cc1 -std=c++2c -fno-contracts -DENABLED=0 -fsyntax-only -verify=disabled %s

// disabled-no-diagnostics

#ifdef __cplusplus
void test() {
#if ENABLED
  int contract_assert; // enabled-error {{expected unqualified-id}}
#else
  int contract_assert;
  ++contract_assert;
#endif
}
#endif
