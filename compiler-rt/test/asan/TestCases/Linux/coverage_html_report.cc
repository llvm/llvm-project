// REQUIRES: has_sancovcc
// REQUIRES: x86_64-linux
// RUN: %clangxx_asan -fsanitize-coverage=func %s -o %t
// RUN: rm -rf %T/coverage_html_report
// RUN: mkdir -p %T/coverage_html_report
// RUN: cd %T/coverage_html_report
// RUN: %env_asan_opts=coverage=1:verbosity=1:html_cov_report=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-main
// RUN: ls *.html | FileCheck %s --check-prefix=CHECK-ls
// RUN: rm -r %T/coverage_html_report

#include <stdio.h>
#include <unistd.h>

void bar() { printf("bar\n"); }

int main(int argc, char **argv) {
  fprintf(stderr, "PID: %d\n", getpid());
  bar();
  return 0;
}

// CHECK-main: PID: [[PID:[0-9]+]]
// CHECK-main: [[PID]].sancov: 2 PCs written
// CHECK-main: html report generated to ./coverage_html_report.cc.tmp.[[PID]].html
// CHECK-ls: coverage_html_report.cc.tmp.{{[0-9]+}}.html
