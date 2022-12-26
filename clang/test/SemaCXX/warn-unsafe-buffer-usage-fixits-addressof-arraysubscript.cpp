<<<<<<< HEAD
// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -triple=arm-apple \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// XFAIL: *

=======
// RUN: %clang_cc1 -triple=arm-apple -std=c++20 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
>>>>>>> 991d7848b740 ([SafeBufferUsage] restore safe buffer usage warnings for MIOpen GTest)
int f(unsigned long, void *);

[[clang::unsafe_buffer_usage]]
int unsafe_f(unsigned long, void *);

void address_to_integer(int x) {
  int * p = new int[10];
  unsigned long n = (unsigned long) &p[5];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:37-[[@LINE-1]]:42}:"&p.data()[5]"
  unsigned long m = (unsigned long) &p[x];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:37-[[@LINE-1]]:42}:"&p.data()[x]"
}

void address_to_bool(int x) {
  int * p = new int[10];
  bool a = (bool) &p[5];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:19-[[@LINE-1]]:24}:"&p.data()[5]"
  bool b = (bool) &p[x];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:19-[[@LINE-1]]:24}:"&p.data()[x]"

  bool a1 = &p[5];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:13-[[@LINE-1]]:18}:"&p.data()[5]"
  bool b1 = &p[x];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:13-[[@LINE-1]]:18}:"&p.data()[x]"

  if (&p[5]) {
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:12}:"&p.data()[5]"
    return;
  }
}

void call_argument(int x) {
  int * p = new int[10];

  f((unsigned long) &p[5], &p[x]);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:21-[[@LINE-1]]:26}:"&p.data()[5]"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:28-[[@LINE-2]]:33}:"&p.data()[x]"
}

void ignore_unsafe_calls(int x) {
  // Cannot fix `&p[x]` for now as it is an argument of an unsafe
  // call. So no fix for variable `p`.
  int * p = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  unsafe_f((unsigned long) &p[5],
	   // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
	   &p[x]);

  int * q = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> q"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"
  unsafe_f((unsigned long) &q[5],
	   // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:28-[[@LINE-1]]:33}:"&q.data()[5]"
	   (void*)0);
}

void odd_subscript_form() {
  int * p = new int[10];
  unsigned long n = (unsigned long) &5[p];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:37-[[@LINE-1]]:42}:"&p.data()[5]"
}

void index_is_zero() {
  int * p = new int[10];
  int n = p[5];

  f((unsigned long)&p[0],
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:20-[[@LINE-1]]:25}:"p.data()"
    &p[0]);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:5-[[@LINE-1]]:10}:"p.data()"
}

void pointer_subtraction(int x) {
  int * p = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> p"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"

  int n = &p[9] - &p[4];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:16}:"&p.data()[9]"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:19-[[@LINE-2]]:24}:"&p.data()[4]"
  if (&p[9] - &p[x]) {
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:12}:"&p.data()[9]"
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:20}:"&p.data()[x]"
    return;
  }
}

// To test multiple function declarations, each of which carries
// different incomplete informations.
// no fix-it in the rest of this test:

[[clang::unsafe_buffer_usage]]
void unsafe_g(void*);

void unsafe_g(void*);

void multiple_unsafe_fundecls() {
  int * p = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  unsafe_g(&p[5]);
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
}

void unsafe_h(void*);

[[clang::unsafe_buffer_usage]]
void unsafe_h(void*);

void unsafe_h(void* p) { ((char*)p)[10]; }

void multiple_unsafe_fundecls2() {
  int * p = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  unsafe_h(&p[5]);
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
}
