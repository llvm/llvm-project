// Test this without pch.
// RUN: %clang_cc1 -include %S/nonvisible-external-defs.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %S/nonvisible-external-defs.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

int g(int, float); // expected-error{{conflicting types}}

// expected-note@nonvisible-external-defs.h:10{{previous declaration}}
