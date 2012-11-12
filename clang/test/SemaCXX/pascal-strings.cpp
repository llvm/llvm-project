// RUN: %clang_cc1 -fsyntax-only -verify %s -fpascal-strings
const wchar_t *pascalString = L"\pThis is a Pascal string";

unsigned char a[3] = "\pa";
unsigned char b[3] = "\pab";
unsigned char c[3] = "\pabc"; // expected-error {{initializer-string for char array is too long}}
