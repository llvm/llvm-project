// RUN: %clangxx_asan -fsanitize-coverage=func,trace-pc-guard -DSHARED %s -shared -o %dynamiclib -fPIC %ld_flags_rpath_so
// RUN: %clangxx_asan -fsanitize-coverage=func,trace-pc-guard %s %ld_flags_rpath_exe -o %t
// RUN: rm -rf %t-dir && mkdir -p %t-dir && cd %t-dir
// RUN: %env_asan_opts=coverage=1:verbosity=1 %run %t 2>&1         | FileCheck %s --check-prefix=CHECK-main
// RUN: %sancov print coverage.*sancov 2>&1 | FileCheck %s --check-prefix=CHECK-SANCOV1
// RUN: %env_asan_opts=coverage=1:verbosity=1 %run %t foo 2>&1     | FileCheck %s --check-prefix=CHECK-foo
// RUN: %sancov print coverage.*sancov 2>&1 | FileCheck %s --check-prefix=CHECK-SANCOV2
// RUN: %env_asan_opts=coverage=1:verbosity=1 %run %t bar 2>&1     | FileCheck %s --check-prefix=CHECK-bar
// RUN: %sancov print coverage.*sancov 2>&1 | FileCheck %s --check-prefix=CHECK-SANCOV2
// RUN: %env_asan_opts=coverage=1:verbosity=1 %run %t foo bar 2>&1 | FileCheck %s --check-prefix=CHECK-foo-bar
// RUN: %sancov print coverage.*sancov 2>&1 | FileCheck %s --check-prefix=CHECK-SANCOV2
// RUN: %sancov print libcoverage.*sancov 2>&1 | FileCheck %s --check-prefix=CHECK-SANCOV1
// RUN: %sancov merge coverage.*sancov > merged-cov
// RUN: %sancov print merged-cov 2>&1 | FileCheck %s --check-prefix=CHECK-SANCOV2
// RUN: %env_asan_opts=coverage=1:verbosity=1 not %run %t foo bar 4    2>&1 | FileCheck %s --check-prefix=CHECK-report
// RUN: %env_asan_opts=coverage=1:verbosity=1 not %run %t foo bar 4 5  2>&1 | FileCheck %s --check-prefix=CHECK-segv
// RUN: cd .. && rm -rf %t-dir
//
// https://code.google.com/p/address-sanitizer/issues/detail?id=263
// XFAIL: android
// XFAIL: i386-netbsd
// UNSUPPORTED: ios

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#ifdef SHARED
void bar() { printf("bar\n"); }
#else
__attribute__((noinline))
void foo() { printf("foo\n"); }
extern void bar();

int G[4];

int main(int argc, char **argv) {
  fprintf(stderr, "PID: %d\n", getpid());
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "foo"))
      foo();
    if (!strcmp(argv[i], "bar"))
      bar();
  }
  if (argc == 5) {
    static volatile char *zero = 0;
    *zero = 0;  // SEGV if argc == 5.
  }
  return G[argc];  // Buffer overflow if argc >= 4.
}
#endif

// CHECK-main: PID: [[PID:[0-9]+]]
// CHECK-main: [[PID]].sancov: 1 PCs written
// CHECK-main-NOT: .so.[[PID]]
//
// CHECK-foo: PID: [[PID:[0-9]+]]
// CHECK-foo: [[PID]].sancov: 2 PCs written
// CHECK-foo-NOT: .so.[[PID]]
//
// CHECK-bar: PID: [[PID:[0-9]+]]
// CHECK-bar-DAG: .so.[[PID]].sancov: 1 PCs written
// CHECK-bar-DAG: [[PID]].sancov: 1 PCs written
//
// CHECK-foo-bar: PID: [[PID:[0-9]+]]
// CHECK-foo-bar-DAG: so.[[PID]].sancov: 1 PCs written
// CHECK-foo-bar-DAG: [[PID]].sancov: 2 PCs written
//
// CHECK-report: AddressSanitizer: global-buffer-overflow
// CHECK-report: PCs written
//
// CHECK-segv: AddressSanitizer: SEGV
// CHECK-segv: PCs written
//
// CHECK-SANCOV1: 1 PCs total
// CHECK-SANCOV2: 2 PCs total
