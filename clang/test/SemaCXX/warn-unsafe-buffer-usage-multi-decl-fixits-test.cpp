// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits -fsafe-buffer-usage-suggestions %s 2>&1 | FileCheck %s

void foo1a() {
  int *r = new int[7];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[4];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  p = r;
  int tmp = p[9];
  int *q;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
  q = r;
}

void foo1b() {
  int *r = new int[7];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[4];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  p = r;
  int tmp = p[9];
  int *q = new int[4];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  q = r;
  tmp = q[9];
}

void foo1c() {
  int *r = new int[7];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[4];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  p = r;
  int tmp = r[9];
  int *q;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
  q = r;
  tmp = q[9];
}

void foo2a() {
  int *r = new int[7];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[5];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 5}"
  int *q = new int[4];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  p = q;
  int tmp = p[8];
  q = r;
}

void foo2b() {
  int *r = new int[7];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[5];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 5}"
  int *q = new int[4];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  p = q;
  int tmp = q[8];
  q = r;
}

void foo2c() {
  int *r = new int[7];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[5];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 5}"
  int *q = new int[4];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  p = q;
  int tmp = p[8];
  q = r;
  tmp = q[8];
}

void foo3a() {
  int *r = new int[7];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  int *p = new int[5];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 5}"
  int *q = new int[4];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
  q = p;
  int tmp = p[8];
  q = r;
}

void foo3b() {
  int *r = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  int *p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  int *q = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  q = p;
  int tmp = q[8];
  q = r;
}
