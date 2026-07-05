// RUN: %clang_cc1 -std=c23 %s -E -verify

#embed "nfejfNejAKFe"
// expected-error@-1 {{'nfejfNejAKFe' file not found}}
