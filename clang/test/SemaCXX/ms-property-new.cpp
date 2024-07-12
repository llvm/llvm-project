// RUN: %clang_cc1 -ast-print -verify -triple=x86_64-pc-win32 -fms-compatibility %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fms-compatibility -emit-pch -o %t %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fms-compatibility -include-pch %t -verify %s -ast-print -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

struct S {
  int GetX() { return 42; }
  __declspec(property(get=GetX)) int x;

  int GetY(int i, int j) { return i+j; }
  __declspec(property(get=GetY)) int y[];

  void* operator new(__SIZE_TYPE__, int);
};

template <typename T>
struct TS {
  T GetT() { return T(); }
  __declspec(property(get=GetT)) T t;

  T GetR(T i, T j) { return i+j; }
  __declspec(property(get=GetR)) T r[];
};

int main(int argc, char **argv) {
  // CHECK: S *s;
  // CHECK-NEXT: new (s->x) S;
  // CHECK-NEXT: new ((s->y)[1][2]) S;
  S *s;
  new (s->x) S;
  new ((s->y)[1][2]) S;

  // CHECK-NEXT: TS<double> *ts;
  // CHECK-NEXT: new (ts->t) S;
  // CHECK-NEXT: new ((ts->r)[1][2]) S;
  TS<double> *ts;
  new (ts->t) S;
  new ((ts->r)[1][2]) S;
}

#endif
