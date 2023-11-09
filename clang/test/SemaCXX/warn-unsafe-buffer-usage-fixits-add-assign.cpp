// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
void foo(int * , int *);

void add_assign_test(int n, int *a) {
  int *p = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  p += 2;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:9}:"p = p.subspan(2)"
  
  int *r = p;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> r"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:13}:", <# placeholder #>}"
  while (*r != 0) {
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:11}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"[0]"
    r += 2;
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:5-[[@LINE-1]]:11}:"r = r.subspan(2)"
  }
  
  if (*p == 0) {
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:8}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:9-[[@LINE-2]]:9}:"[0]"
    p += n;
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:5-[[@LINE-1]]:11}:"p = p.subspan(n)"
  }
  
  if (*p == 1)
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:8}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:9-[[@LINE-2]]:9}:"[0]"
    p += 3;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:5-[[@LINE-1]]:11}:"p = p.subspan(3)"
  
  a += -9;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:5-[[@LINE-1]]:11}:"p = p.subspan(-9)"
}
