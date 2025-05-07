
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

void test(void) {
  // CHECK: VarDecl {{.+}} p1 '_Atomic(int *__unsafe_indexable)'
  int *_Atomic __unsafe_indexable p1;
  // CHECK: VarDecl {{.+}} p2 '_Atomic(int *__unsafe_indexable)'
  int *__unsafe_indexable _Atomic p2;
  // CHECK: VarDecl {{.+}} p3 '_Atomic(int *__unsafe_indexable)'
  _Atomic(int *__unsafe_indexable) p3;
  // CHECK: VarDecl {{.+}} p4 '_Atomic(int *__unsafe_indexable)'
  _Atomic(int *) __unsafe_indexable p4;

  // CHECK: VarDecl {{.+}} p5 '_Atomic(int *__single)'
  int *_Atomic __single p5;
  // CHECK: VarDecl {{.+}} p6 '_Atomic(int *__single)'
  int *__single _Atomic p6;
  // CHECK: VarDecl {{.+}} p7 '_Atomic(int *__single)'
  _Atomic(int *__single) p7;
  // CHECK: VarDecl {{.+}} p8 '_Atomic(int *__single)'
  _Atomic(int *) __single p8;

  // CHECK: VarDecl {{.+}} p9 '_Atomic(_Atomic(int *__unsafe_indexable) *__unsafe_indexable)'
  int *_Atomic __unsafe_indexable *_Atomic __unsafe_indexable p9;
  // CHECK: VarDecl {{.+}} p10 '_Atomic(_Atomic(int *__unsafe_indexable) *__unsafe_indexable)'
  int *__unsafe_indexable _Atomic *_Atomic __unsafe_indexable p10;
  // CHECK: VarDecl {{.+}} p11 '_Atomic(_Atomic(int *__unsafe_indexable) *__unsafe_indexable)'
  _Atomic(int *__unsafe_indexable) *_Atomic __unsafe_indexable p11;
  // CHECK: VarDecl {{.+}} p12 '_Atomic(_Atomic(int *__unsafe_indexable) *__unsafe_indexable)'
  _Atomic(int *) __unsafe_indexable *_Atomic __unsafe_indexable p12;

  // CHECK: VarDecl {{.+}} p13 '_Atomic(_Atomic(int *__single) *__unsafe_indexable)'
  int *_Atomic __single *_Atomic __unsafe_indexable p13;
  // CHECK: VarDecl {{.+}} p14 '_Atomic(_Atomic(int *__single) *__unsafe_indexable)'
  int *__single _Atomic *_Atomic __unsafe_indexable p14;
  // CHECK: VarDecl {{.+}} p15 '_Atomic(_Atomic(int *__single) *__unsafe_indexable)'
  _Atomic(int *__single) *_Atomic __unsafe_indexable p15;
  // CHECK: VarDecl {{.+}} p16 '_Atomic(_Atomic(int *__single) *__unsafe_indexable)'
  _Atomic(int *) __single *_Atomic __unsafe_indexable p16;

  // CHECK: VarDecl {{.+}} p17 '_Atomic(_Atomic(int *__unsafe_indexable) *__single)'
  int *_Atomic __unsafe_indexable *_Atomic __single p17;
  // CHECK: VarDecl {{.+}} p18 '_Atomic(_Atomic(int *__unsafe_indexable) *__single)'
  int *__unsafe_indexable _Atomic *_Atomic __single p18;
  // CHECK: VarDecl {{.+}} p19 '_Atomic(_Atomic(int *__unsafe_indexable) *__single)'
  _Atomic(int *__unsafe_indexable) *_Atomic __single p19;
  // CHECK: VarDecl {{.+}} p20 '_Atomic(_Atomic(int *__unsafe_indexable) *__single)'
  _Atomic(int *) __unsafe_indexable *_Atomic __single p20;

  // CHECK: VarDecl {{.+}} p21 '_Atomic(_Atomic(int *__single) *__single)'
  int *_Atomic __single *_Atomic __single p21;
  // CHECK: VarDecl {{.+}} p22 '_Atomic(_Atomic(int *__single) *__single)'
  int *__single _Atomic *_Atomic __single p22;
  // CHECK: VarDecl {{.+}} p23 '_Atomic(_Atomic(int *__single) *__single)'
  _Atomic(int *__single) *_Atomic __single p23;
  // CHECK: VarDecl {{.+}} p24 '_Atomic(_Atomic(int *__single) *__single)'
  _Atomic(int *) __single *_Atomic __single p24;
}
