// RUN: %clang_cc1 -triple wasm32-unknown-unknown %s -fsyntax-only -verify
// RUN: %clang_cc1 -triple wasm64-unknown-unknown %s -fsyntax-only -verify

void __attribute__((regparm(2))) fc_f1(int i, int j, int k) {} // expected-error{{'regparm' is not valid on this platform}}
