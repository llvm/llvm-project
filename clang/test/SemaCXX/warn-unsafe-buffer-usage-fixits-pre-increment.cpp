// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void foo(int * , int *);

void simple() {
  int * p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"
  bool b = ++p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:12-[[@LINE-1]]:15}:"(p = p.subspan(1)).data()"
  unsigned long n = (unsigned long) ++p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:37-[[@LINE-1]]:40}:"(p = p.subspan(1)).data()"
  if (++p) {
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:10}:"(p = p.subspan(1)).data()"
  }
  if (++p - ++p) {
    // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:10}:"(p = p.subspan(1)).data()"
    // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:16}:"(p = p.subspan(1)).data()"
  }
  foo(++p, p);
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:10}:"(p = p.subspan(1)).data()"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:".data()"

  // FIXME: Don't know how to fix the following cases:
  // CHECK-NOT: fix-it:"{{.*}}":{
  int * g = new int[10];
  int * a[10];
  a[0] = ++g;
  foo(g++, g);
  if (++(a[0])) {
  }
}
