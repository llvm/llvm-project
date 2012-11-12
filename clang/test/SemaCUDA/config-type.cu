// RUN: %clang_cc1 -fsyntax-only -verify %s

void cudaConfigureCall(unsigned gridSize, unsigned blockSize); // expected-error {{must have scalar return type}}
