// RUN: %clang_cc1 -fno-rtti -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only -mms-bitfields -verify %s 2>&1

struct A {
  char a : 9; // expected-error{{width of bit-field 'a' (9 bits) exceeds the size of its type (8 bits)}}
  int b : 33; // expected-error{{width of bit-field 'b' (33 bits) exceeds the size of its type (32 bits)}}
  bool c : 9; // expected-error{{width of bit-field 'c' (9 bits) exceeds the size of its type (8 bits)}}
  bool d : 3;
};

int a[sizeof(A) == 1 ? 1 : -1];
