// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
//
// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 -mllvm -dfsan-instrument-with-call-threshold=0 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
//
// REQUIRES: x86_64-target-arch

#include <sanitizer/dfsan_interface.h>
#include <stdio.h>
#include <string.h>

__attribute__((noinline)) int foo(int a, int b) { return a + b; }

__attribute__((noinline)) void bar(int depth, void *addr, int size) {
  if (depth) {
    bar(depth - 1, addr, size);
  } else {
    dfsan_set_label(1, addr, size);
  }
}

__attribute__((noinline)) void baz(int depth, void *addr, int size) {
  bar(depth, addr, size);
}

int main(int argc, char *argv[]) {
  int a = 10;
  int b = 20;
  baz(8, &a, sizeof(a));
  int c = foo(a, b);
  dfsan_origin c_o = dfsan_get_origin(c);
  dfsan_print_origin_id_trace(c_o);
  // CHECK: Origin value: {{.*}}, Taint value was created at
  // CHECK: #0 {{.*}} in bar.dfsan {{.*}}origin_id_stack_trace.c:[[@LINE-16]]
  // CHECK-COUNT-8: #{{[0-9]+}} {{.*}} in bar.dfsan {{.*}}origin_id_stack_trace.c:[[@LINE-19]]
  // CHECK: #9 {{.*}} in baz.dfsan {{.*}}origin_id_stack_trace.c:[[@LINE-13]]

  char buf[3000];
  size_t length = dfsan_sprint_origin_id_trace(c_o, buf, sizeof(buf));

  printf("==OUTPUT==\n\n%s==EOS==\n", buf);
  // CHECK: ==OUTPUT==
  // CHECK: Origin value: {{.*}}, Taint value was created at
  // CHECK: #0 {{.*}} in bar.dfsan {{.*}}origin_id_stack_trace.c:[[@LINE-26]]
  // CHECK-COUNT-8: #{{[0-9]+}} {{.*}} in bar.dfsan {{.*}}origin_id_stack_trace.c:[[@LINE-29]]
  // CHECK: #9 {{.*}} in baz.dfsan {{.*}}origin_id_stack_trace.c:[[@LINE-23]]
  // CHECK: ==EOS==

  char tinybuf[20];
  size_t same_length =
      dfsan_sprint_origin_id_trace(c_o, tinybuf, sizeof(tinybuf));

  printf("==TRUNCATED OUTPUT==\n\n%s==EOS==\n", tinybuf);
  // CHECK: ==TRUNCATED OUTPUT==
  // CHECK: Origin value: 0x1==EOS==

  printf("Returned length: %zu\n", length);
  printf("Actual length: %zu\n", strlen(buf));
  printf("Returned length with truncation: %zu\n", same_length);

  // CHECK: Returned length: [[#LEN:]]
  // CHECK: Actual length: [[#LEN]]
  // CHECK: Returned length with truncation: [[#LEN]]

  buf[0] = '\0';
  length = dfsan_sprint_origin_id_trace(c_o, buf, 0);
  printf("Output=\"%s\"\n", buf);
  printf("Returned length: %zu\n", length);
  // CHECK: Output=""
  // CHECK: Returned length: [[#LEN]]
}
