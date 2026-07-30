// RUN: %clang_cc1 %s -verify -fsyntax-only

int p1 __attribute__((endian(host)));	// expected-warning {{unknown attribute 'endian' ignored}}
