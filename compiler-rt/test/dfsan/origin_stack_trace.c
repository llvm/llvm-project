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

#define NOINLINE __attribute__((noinline))

NOINLINE int foo(int a, int b) { return a + b; }

NOINLINE void bar(int depth, void *addr, int size) {
  if (depth) {
    bar(depth - 1, addr, size);
  } else {
    dfsan_set_label(1, addr, size);
  }
}

NOINLINE void baz(int depth, void *addr, int size) {
  bar(depth, addr, size);
}

int main(int argc, char *argv[]) {
  int a = 10;
  int b = 20;
  baz(8, &a, sizeof(a));
  int c = foo(a, b);
  dfsan_print_origin_trace(&c, NULL);
  // CHECK: Taint value 0x1 {{.*}} origin tracking ()
  // CHECK: Origin value: {{.*}}, Taint value was stored to memory at
  // CHECK: #0 {{.*}} in main {{.*}}origin_stack_trace.c:[[@LINE-4]]

  // CHECK: Origin value: {{.*}}, Taint value was created at
  // CHECK: #0 {{.*}} in bar.dfsan {{.*}}origin_stack_trace.c:[[@LINE-19]]
  // CHECK-COUNT-8: #{{[0-9]+}} {{.*}} in bar.dfsan {{.*}}origin_stack_trace.c:[[@LINE-22]]
  // CHECK: #9 {{.*}} in baz.dfsan {{.*}}origin_stack_trace.c:[[@LINE-16]]

  char buf[3000];
  size_t length = dfsan_sprint_origin_trace(&c, NULL, buf, sizeof(buf));

  printf("==OUTPUT==\n\n%s==EOS==\n", buf);
  // CHECK: ==OUTPUT==
  // CHECK: Taint value 0x1 {{.*}} origin tracking ()
  // CHECK: Origin value: {{.*}}, Taint value was stored to memory at
  // CHECK: #0 {{.*}} in main {{.*}}origin_stack_trace.c:[[@LINE-18]]

  // CHECK: Origin value: {{.*}}, Taint value was created at
  // CHECK: #0 {{.*}} in bar.dfsan {{.*}}origin_stack_trace.c:[[@LINE-33]]
  // CHECK-COUNT-8: #{{[0-9]+}} {{.*}} in bar.dfsan {{.*}}origin_stack_trace.c:[[@LINE-36]]
  // CHECK: #9 {{.*}} in baz.dfsan {{.*}}origin_stack_trace.c:[[@LINE-30]]
  // CHECK: ==EOS==

  char tinybuf[18];
  size_t same_length = dfsan_sprint_origin_trace(&c, NULL, tinybuf, sizeof(tinybuf));

  printf("==TRUNCATED OUTPUT==\n\n%s==EOS==\n", tinybuf);
  // CHECK: ==TRUNCATED OUTPUT==
  // CHECK: Taint value 0x1==EOS==

  printf("Returned length: %zu\n", length);
  printf("Actual length: %zu\n", strlen(buf));
  printf("Returned length with truncation: %zu\n", same_length);

  // CHECK: Returned length: [[#LEN:]]
  // CHECK: Actual length: [[#LEN]]
  // CHECK: Returned length with truncation: [[#LEN]]

  size_t length_with_desc = dfsan_sprint_origin_trace(&c, "DESCRIPTION", buf, sizeof(buf));

  printf("==OUTPUT==\n\n%s==EOS==\n", buf);
  // CHECK: ==OUTPUT==
  // CHECK: Taint value 0x1 {{.*}} origin tracking (DESCRIPTION)
  // CHECK: Origin value: {{.*}}, Taint value was stored to memory at
  // CHECK: #0 {{.*}} in main {{.*}}origin_stack_trace.c:[[@LINE-47]]

  // CHECK: Origin value: {{.*}}, Taint value was created at
  // CHECK: #0 {{.*}} in bar.dfsan {{.*}}origin_stack_trace.c:[[@LINE-62]]
  // CHECK-COUNT-8: #{{[0-9]+}} {{.*}} in bar.dfsan {{.*}}origin_stack_trace.c:[[@LINE-65]]
  // CHECK: #9 {{.*}} in baz.dfsan {{.*}}origin_stack_trace.c:[[@LINE-59]]
  // CHECK: ==EOS==

  printf("Returned length: %zu\n", length_with_desc);
  // COMM: Message length is increased by 11: the length of "DESCRIPTION".
  // CHECK: Returned length: [[#LEN + 11]]

  buf[0] = '\0';
  length = dfsan_sprint_origin_trace(&c, NULL, buf, 0);
  printf("Output=\"%s\"\n", buf);
  printf("Returned length: %zu\n", length);
  // CHECK: Output=""
  // CHECK: Returned length: [[#LEN]]
}
