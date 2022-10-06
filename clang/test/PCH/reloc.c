// RUN: %clang -target x86_64-apple-darwin11 --relocatable-pch -o %t \
// RUN:   -isysroot %S/Inputs/libroot %S/Inputs/libroot/usr/include/reloc.h
// RUN: %clang -target x86_64-apple-darwin11 -fsyntax-only \
// RUN:   -include-pch %t -isysroot %S/Inputs/libroot %s -Xclang -verify
// RUN: not %clang -target x86_64-apple-darwin11 -include-pch %t %s
// REQUIRES: x86-registered-target

#include <reloc.h>

int x = 2; // expected-error{{redefinition}}
int y = 5; // expected-error{{redefinition}}


// expected-note@Inputs/libroot/usr/include/reloc.h:13{{previous definition}}
// expected-note@Inputs/libroot/usr/include/reloc2.h:14{{previous definition}}
