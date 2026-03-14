// RUN: %clang_cc1 -verify %s 
constant int no_init; // expected-error {{variable in constant address space must be initialized}}
