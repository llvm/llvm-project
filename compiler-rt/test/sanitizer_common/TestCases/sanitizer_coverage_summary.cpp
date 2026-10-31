// Tests print_coverage_summary for SanitizerCoverage dumps.

// REQUIRES: has_sancovcc
// UNSUPPORTED: ubsan,i386-darwin,target={{(powerpc64|s390x|sparc|thumb).*}}
// This test is failing for lsan on darwin on x86_64h.
// UNSUPPORTED: x86_64h-darwin && lsan
// XFAIL: tsan
// XFAIL: android && asan
// XFAIL: darwin-remote
// UNSUPPORTED: rtsan

// RUN: rm -rf %t_workdir
// RUN: mkdir -p %t_workdir
// RUN: cd %t_workdir
// RUN: %clangxx -O0 -fsanitize-coverage=trace-pc-guard %s -o %t
// RUN: %env_tool_opts=coverage=1 %t 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: rm -f *.sancov
// RUN: %env_tool_opts=coverage=1:print_coverage_summary=0 %t 2>&1 | FileCheck %s --check-prefix=CHECK-QUIET --implicit-check-not='SanitizerCoverage'
// RUN: ls *.sancov
// RUN: rm -rf %t_workdir

#include <stdio.h>

int main() {
  fprintf(stderr, "main\n");
  return 0;
}

// CHECK-DEFAULT: main
// CHECK-DEFAULT: SanitizerCoverage: {{.*}}.sancov: {{[0-9]+}} PCs written

// CHECK-QUIET: main
