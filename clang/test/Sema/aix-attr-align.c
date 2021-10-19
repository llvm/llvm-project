// off-no-diagnostics
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -verify=off -Wno-aix-compat -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -verify=off -Wno-aix-compat -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux -verify=off -fsyntax-only %s

struct S {
  int a[8] __attribute__((aligned(8))); // no-warning
};

struct T {
  int a[4] __attribute__((aligned(16))); // expected-warning {{requesting an alignment of 16 bytes or greater for struct members is not binary compatible with AIX XL 16.1 and older}}
};

struct U {
  int a[2] __attribute__((aligned(32))); // expected-warning {{requesting an alignment of 16 bytes or greater for struct members is not binary compatible with AIX XL 16.1 and older}}
};

int a[8] __attribute__((aligned(8)));  // no-warning
int b[4] __attribute__((aligned(16))); // no-warning
int c[2] __attribute__((aligned(32))); // no-warning
