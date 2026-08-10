// RUN: %clang_cc1 -verify %s

constant long a __attribute__((endian(host))) = 100;	// expected-warning {{unknown attribute 'endian' ignored}}
