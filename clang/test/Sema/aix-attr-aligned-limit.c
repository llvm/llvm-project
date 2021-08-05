// RUN: %clang_cc1 -triple powerpc-unknown-aix -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fsyntax-only -verify %s
//
int a __attribute__((aligned(8192))); // expected-error {{requested alignment must be 4096 bytes or smaller}}
