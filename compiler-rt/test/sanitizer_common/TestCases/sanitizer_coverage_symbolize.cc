// Tests trace pc guard coverage collection.
//
// REQUIRES: has_sancovcc,stable-runtime
// XFAIL: tsan,darwin
// TODO: this will fail on powerpc64,s390x once sancov will be called.
//
// RUN: DIR=%t_workdir
// RUN: rm -rf $DIR
// RUN: mkdir -p $DIR
// RUN: cd $DIR
// RUN: %clangxx -O0 -fsanitize-coverage=trace-pc-guard %s -ldl -o %t
// RUN: %env_tool_opts=coverage=1 %t 2>&1 | FileCheck %s
// RUN: %env_tool_opts=coverage=1 SANCOV_OPTIONS=symbolize=0 %t 2>&1 | FileCheck %s --check-prefix=CHECK-NOSYM
// RUN: rm -rf $DIR

#include <stdio.h>

int foo() {
  fprintf(stderr, "foo\n");
  return 1;
}

int main() {
  fprintf(stderr, "main\n");
  foo();
  foo();
}

// CHECK: main
// CHECK: SanitizerCoverage: ./sanitizer_coverage_symbolize.{{.*}}.sancov 2 PCs written
// CHECK: call sancov

// CHECK-NOSYM: main
// CHECK-NOSYM: SanitizerCoverage: ./sanitizer_coverage_symbolize.{{.*}}.sancov 2 PCs written
// CHECK-NOSYM-NOT: call sancov
