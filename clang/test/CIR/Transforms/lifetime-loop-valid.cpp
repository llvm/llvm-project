// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-tool %t.cir -cir-lifetime-check="history=invalid,null remarks=pset-always" -verify-diagnostics -o %t-out.cir
// XFAIL: *

//
// Loops that do not change psets

// p1179r1: 2.4.9.1
// No diagnostic needed, pset(p) = {a} before and after the loop
void valid0(bool b, int j) {
  int a[10];
  int *p = &a[0];
  while (j) {
    if (b) {
      p = &a[j];
    }
    j = j - 1;
  }
  *p = 12; // expected-remark {{pset => { a }}}
}

// p1179r1: 2.4.9.2
void valid1(bool b, int j) {
  int a[4], c[5];
  int *p = &a[0];
  while (j) {
    if (b) {
      p = &c[j];
    }
    j = j - 1;
  }
  *p = 0; // expected-remark {{pset => { a, c }}}

  while (j) {
    if (b) {
      p = &c[j];
    }
    j = j - 1;
  }
  *p = 0; // expected-remark {{pset => { a, c }}}
}