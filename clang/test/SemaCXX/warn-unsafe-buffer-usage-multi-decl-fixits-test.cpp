// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits -fsafe-buffer-usage-suggestions %s 2>&1 | FileCheck %s

void foo1a() {
  int *r = new int[7];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int *p = new int[4];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  p = r;
  int tmp = p[9];
  int *q;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  q = r;  // FIXME: we do not fix `q = r` here as the `.data()` fix-it is not generally correct
}

void foo1b() {
  int *r = new int[7];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[4];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  p = r;
  int tmp = p[9];
  int *q = new int[4];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  q = r;
  tmp = q[9];
}

void foo1c() {
  int *r = new int[7];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int *p = new int[4];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  p = r;  // FIXME: we do not fix `p = r` here as the `.data()` fix-it is not generally correct
  int tmp = r[9];
  int *q;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  q = r;
  tmp = q[9];
}

void foo2a() {
  int *r = new int[7];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[5];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 5}"
  int *q = new int[4];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  p = q;
  int tmp = p[8];
  q = r;
}

void foo2b() {
  int *r = new int[7];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int *p = new int[5];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int *q = new int[4];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  p = q;  // FIXME: we do not fix `p = q` here as the `.data()` fix-it is not generally correct
  int tmp = q[8];
  q = r;
}

void foo2c() {
  int *r = new int[7];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 7}"
  int *p = new int[5];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 5}"
  int *q = new int[4];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", 4}"
  p = q;
  int tmp = p[8];
  q = r;
  tmp = q[8];
}

void foo3a() {
  int *r = new int[7];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int *p = new int[5];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int *q = new int[4];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  q = p;  // FIXME: we do not fix `q = p` here as the `.data()` fix-it is not generally correct
  int tmp = p[8];
  q = r;
}

void foo3b() {
  int *r = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  int *p = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  int *q = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  q = p;
  int tmp = q[8];
  q = r;
}
