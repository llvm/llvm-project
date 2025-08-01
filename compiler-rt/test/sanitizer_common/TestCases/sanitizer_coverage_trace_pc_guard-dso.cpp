// Tests trace pc guard coverage collection.

// REQUIRES: has_sancovcc
// Doesn't work on big-endian targets.
// UNSUPPORTED: ubsan,target={{(powerpc64|s390x|sparc|thumb).*}}
// XFAIL: tsan,darwin
// XFAIL: android && asan

// RUN: rm -rf %t_workdir
// RUN: mkdir -p %t_workdir
// RUN: cd %t_workdir
// RUN: %clangxx -DSHARED1 -O0 -fsanitize-coverage=trace-pc-guard -shared %s -o %t_1.so -fPIC
// RUN: %clangxx -DSHARED2 -O0 -fsanitize-coverage=trace-pc-guard -shared %s -o %t_2.so -fPIC
// RUN: %clangxx -DMAIN -O0 -fsanitize-coverage=trace-pc-guard %s -o %t %t_1.so %t_2.so
// RUN: %env_tool_opts=coverage=1 %t 2>&1 | FileCheck %s
// RUN: %sancovcc  -covered-functions -strip_path_prefix=TestCases/ *.sancov \
// RUN:            %t %t_1.so %t_2.so 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-SANCOV %s
// RUN: rm -rf %t_workdir

#include <stdio.h>

extern "C" {
  int bar();
  int baz();
}

#ifdef MAIN

int foo() {
  fprintf(stderr, "foo\n");
  return 1;
}

int main() {
  fprintf(stderr, "main\n");
  foo();
  bar();
  baz();
}

#endif // MAIN

extern "C" {

#ifdef SHARED1
int bar() {
  fprintf(stderr, "bar\n");
  return 1;
}
#endif

#ifdef SHARED2
int baz() {
  fprintf(stderr, "baz\n");
  return 1;
}
#endif

} // extern "C"

// CHECK: main
// CHECK-NEXT: foo
// CHECK-NEXT: bar
// CHECK-NEXT: baz
// CHECK-DAG: SanitizerCoverage: ./sanitizer_coverage_trace_pc_guard-dso.{{.*}}.sancov: 2 PCs written
// CHECK-DAG: SanitizerCoverage: ./sanitizer_coverage_trace_pc_guard-dso.{{.*}}_2.so.{{.*}}.sancov: 1 PCs written
// CHECK-DAG: SanitizerCoverage: ./sanitizer_coverage_trace_pc_guard-dso.{{.*}}_1.so.{{.*}}.sancov: 1 PCs written
//
// CHECK-SANCOV: Ignoring {{.*}}_1.so and its coverage because __sanitizer_cov* functions were not found.
// CHECK-SANCOV: Ignoring {{.*}}_2.so and its coverage because __sanitizer_cov* functions were not found.
// CHECK-SANCOV-NEXT: sanitizer_coverage_trace_pc_guard-dso.cpp:[[@LINE-42]] foo
// CHECK-SANCOV-NEXT: sanitizer_coverage_trace_pc_guard-dso.cpp:[[@LINE-38]] main
