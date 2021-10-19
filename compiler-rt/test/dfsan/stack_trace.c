// RUN: %clang_dfsan -gmlt %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
//
// REQUIRES: x86_64-target-arch

#include <assert.h>
#include <sanitizer/dfsan_interface.h>
#include <stdio.h>
#include <string.h>

#define NOINLINE __attribute__((noinline))

NOINLINE size_t bar(int depth, char *buf, size_t len) {
  if (!depth) {
    return dfsan_sprint_stack_trace(buf, len);
  }

  return bar(depth - 1, buf, len);
}

NOINLINE size_t baz(int depth, char *buf, size_t len) {
  return bar(depth, buf, len);
}

int main(int argc, char *argv[]) {
  char buf[3000];
  size_t length = dfsan_sprint_stack_trace(buf, sizeof(buf));
  assert(length < sizeof(buf));
  printf("==OUTPUT==\n%s==EOS==\n", buf);

  // CHECK: ==OUTPUT==
  // CHECK: #0 {{.*}} in main [[FILEPATH:.*]]/stack_trace.c:[[# @LINE - 5 ]]
  // CHECK: ==EOS==

  length = baz(8, buf, sizeof(buf));
  printf("==OUTPUT==\n%s==EOS==\n", buf);

  // CHECK: ==OUTPUT==
  // CHECK: #0 {{.*}} in bar.dfsan [[FILEPATH]]/stack_trace.c:15
  // CHECK-COUNT-8: #{{[1-9]+}} {{.*}} in bar.dfsan [[FILEPATH]]/stack_trace.c:18
  // CHECK: #9 {{.*}} in baz.dfsan [[FILEPATH]]/stack_trace.c:22
  // CHECK: #10 {{.*}} in main [[FILEPATH]]/stack_trace.c:[[# @LINE - 7 ]]
  // CHECK: ==EOS==

  char tinybuf[8];
  size_t same_length = baz(8, tinybuf, sizeof(tinybuf));

  printf("==TRUNCATED OUTPUT==\n%s==EOS==\n", tinybuf);
  // CHECK: ==TRUNCATED OUTPUT==
  // CHECK:     #0 ==EOS==

  printf("Returned length: %zu\n", length);
  printf("Actual length: %zu\n", strlen(buf));
  printf("Returned length with truncation: %zu\n", same_length);

  // CHECK: Returned length: [[#LEN:]]
  // CHECK: Actual length: [[#LEN]]
  // CHECK: Returned length with truncation: [[#LEN]]

  buf[0] = '\0';
  length = baz(8, buf, 0);
  printf("Output=\"%s\"\n", buf);
  printf("Returned length: %zu\n", length);
  // CHECK: Output=""
  // CHECK: Returned length: [[#LEN]]
}
