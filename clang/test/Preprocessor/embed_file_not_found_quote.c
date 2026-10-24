// RUN: %clang_cc1 -std=c23 %s -E -verify
// RUN: %clang_cc1 -std=c23 %s -E -dE -verify | FileCheck %s

// expected-error@+1 {{'nfejfNejAKFe' file not found}}
#embed "nfejfNejAKFe"
int after;

// CHECK: #embed "nfejfNejAKFe" /* clang -E -dE */
// CHECK-NEXT: int after;
