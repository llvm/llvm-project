// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.NewDelete,core,alpha.core.CallAndMessageUnInitRefArg -analyzer-output=text -verify %s
// Passing uninitialized const data to unknown function

#include "Inputs/system-header-simulator-cxx.h"

void doStuff6(const int& c);
void doStuff4(const int y);
void doStuff3(int& g);
void doStuff_uninit(const int *u);


int f10(void) {
  int *ptr;

  ptr = new int; //
  if(*ptr) {
    doStuff4(*ptr);
  }
  delete ptr;
  return 0;
}

int f9(void) {
  int *ptr;

  ptr = new int; //

  doStuff_uninit(ptr); // no warning
  delete ptr;
  return 0;
}

int f8(void) {
  int *ptr;

  ptr = new int;
  *ptr = 25;

  doStuff_uninit(ptr); // no warning?
  delete ptr;
  return 0;
}

void f7(void) {
  int m = 3;
  doStuff6(m); // no warning
}


int& f6_1_sub(int &p) {
  return p;
}

void f6_1(void) {
  int t; // expected-note{{'t' declared without an initial value}}
  int p = f6_1_sub(t); //expected-warning {{Assigned value is garbage or undefined}}
                       //expected-note@-1 {{Calling 'f6_1_sub'}}
                       //expected-note@-2 {{Returning from 'f6_1_sub'}}
                       //expected-note@-3 {{Assigned value is garbage or undefined}}
  int q = p;
  doStuff6(q);
}

void f6_2(void) {
  int t;       //expected-note {{'t' declared without an initial value}}
  int &p = t;
  int &s = p;
  int &q = s;  //expected-note {{'q' initialized here}}
  doStuff6(q); //expected-warning {{1st function call argument is an uninitialized value}}
               //expected-note@-1 {{1st function call argument is an uninitialized value}}
}

void doStuff6_3(int& q_, int *ptr_) {}

void f6_3(void) {
  int *ptr;    //expected-note {{'ptr' declared without an initial value}}
  int t;
  int &p = t;
  int &s = p;
  int &q = s;
  doStuff6_3(q,ptr); //expected-warning {{2nd function call argument is an uninitialized value}}
               //expected-note@-1 {{2nd function call argument is an uninitialized value}}

}

void f6(void) {
  int k;       // expected-note {{'k' declared without an initial value}}
  doStuff6(k); // expected-warning {{1st function call argument is an uninitialized value}}
               // expected-note@-1 {{1st function call argument is an uninitialized value}}

}



void f5(void) {
  int t;
  int* tp = &t;        // expected-note {{'tp' initialized here}}
  doStuff_uninit(tp);  // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                       // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}


void f4(void) {
      int y;        // expected-note {{'y' declared without an initial value}}
      doStuff4(y);  // expected-warning {{1st function call argument is an uninitialized value}}
                    // expected-note@-1 {{1st function call argument is an uninitialized value}}
}

void f3(void) {
      int g;
      doStuff3(g); // no warning
}

int z;
void f2(void) {
      doStuff_uninit(&z);  // no warning
}

void f1(void) {
      int x_=5;
      doStuff_uninit(&x_);  // no warning
}

void f_uninit(void) {
      int x;               // expected-note {{'x' declared without an initial value}}
      doStuff_uninit(&x);  // expected-warning {{1st function call argument is a pointer to uninitialized value}}
                           // expected-note@-1 {{1st function call argument is a pointer to uninitialized value}}
}
