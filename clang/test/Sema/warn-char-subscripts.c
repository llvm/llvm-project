// RUN: %clang_cc1 -Wchar-subscripts -fsyntax-only -verify %s

void t1(void) {
  int array[1] = { 0 };
  char subscript = 0;
  int val = array[subscript]; // expected-warning{{array subscript is of type 'char'}}
}

void t2(void) {
  int array[1] = { 0 };
  char subscript = 0;
  int val = subscript[array]; // expected-warning{{array subscript is of type 'char'}}
}

void t3(void) {
  int *array = 0;
  char subscript = 0;
  int val = array[subscript]; // expected-warning{{array subscript is of type 'char'}}
}

void t4(void) {
  int *array = 0;
  char subscript = 0;
  int val = subscript[array]; // expected-warning{{array subscript is of type 'char'}}
}

char returnsChar(void);
void t5(void) {
  int *array = 0;
  int val = array[returnsChar()]; // expected-warning{{array subscript is of type 'char'}}
}

void t6(void) {
  int array[1] = { 0 };
  signed char subscript = 0;
  int val = array[subscript]; // no warning for explicit signed char
}

void t7(void) {
  int array[1] = { 0 };
  unsigned char subscript = 0;
  int val = array[subscript]; // no warning for unsigned char
}

typedef char CharTy;
void t8(void) {
  int array[1] = { 0 };
  CharTy subscript = 0;
  int val = array[subscript]; // expected-warning{{array subscript is of type 'char'}}
}

typedef signed char SignedCharTy;
void t9(void) {
  int array[1] = { 0 };
  SignedCharTy subscript = 0;
  int val = array[subscript]; // no warning for explicit signed char
}

typedef unsigned char UnsignedCharTy;
void t10(void) {
  int array[1] = { 0 };
  UnsignedCharTy subscript = 0;
  int val = array[subscript]; // no warning for unsigned char
}
