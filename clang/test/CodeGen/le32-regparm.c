// RUN: %clang_cc1 -triple le32-unknown-nacl %s -fsyntax-only -verify
// RUN: %clang_cc1 -triple aarch64 %s -fsyntax-only -verify

void __attribute__((regparm(2))) fc_f1(int i, int j, int k) {} // expected-error{{'regparm' is not valid on this platform}}

