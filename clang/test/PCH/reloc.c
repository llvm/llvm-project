// RUN: %clang --relocatable-pch -o %t -isysroot %S/libroot %S/libroot/usr/include/reloc.h
// RUN: %clang -fsyntax-only -include-pch %t -isysroot %S/libroot %s -Xclang -verify
// RUN: not %clang -include-pch %t %s

#include <reloc.h>

int x = 2; // expected-error{{redefinition}}
int y = 5; // expected-error{{redefinition}}




// expected-note{{previous definition}}
// expected-note{{previous definition}}
