// RUN: %clangxx_asan -std=c++17 -O0 -fsanitize=alloc-token -falloc-token-max=1 %s -o %t
// RUN: not %run %t 0 2>&1 | FileCheck %s --check-prefixes=CHECK,UAF
// RUN: not %run %t 1 2>&1 | FileCheck %s --check-prefixes=CHECK,UAF
// RUN: not %run %t 2 2>&1 | FileCheck %s --check-prefixes=CHECK,OVERFLOW
// RUN: not %run %t 3 2>&1 | FileCheck %s --check-prefix=PLAIN
// RUN: %if x86_64-target-arch %{ %env_asan_opts=detect_leaks=0:quarantine_size_mb=0:thread_local_quarantine_size_kb=0 %run %t 4 2>&1 | FileCheck %s --check-prefix=REUSE %}
// RUN: %clangxx_asan -std=c++17 -O2 -fsanitize=alloc-token -falloc-token-max=1 %s -o %t
// RUN: not %run %t 0 2>&1 | FileCheck %s --check-prefixes=CHECK,UAF
// RUN: not %run %t 1 2>&1 | FileCheck %s --check-prefixes=CHECK,UAF
// REQUIRES: linux, (x86_64-target-arch || aarch64-target-arch)
// UNSUPPORTED: android

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <sanitizer/asan_interface.h>

struct alignas(64) Aligned {
  int value;
};

static void report_callback(const char *report) {
  assert(strstr(report, "Allocation token: 0x0"));
  fprintf(stderr, "Token in report buffer\n");
}

__attribute__((noinline, no_sanitize("alloc-token"))) void *
ordinary_allocation() {
  return malloc(40);
}

int main(int argc, char **argv) {
  int kind = atoi(argv[1]);
  if (kind == 4) {
    // Both requests use the same x86-64 size class.
    void *p = malloc(32);
    __asan_describe_address(p);
    free(p);
    void *q = ordinary_allocation();
    assert(p == q);
    memset(q, 1, 40);
    __asan_describe_address(q);
    free(q);
    return 0;
  }
  if (kind != 3)
    __asan_set_error_report_callback(report_callback);
  if (kind == 1) {
    volatile Aligned *p = new (std::nothrow) Aligned[2];
    delete[] p;
    p->value = 1;
  } else {
    volatile int *p =
        static_cast<int *>(kind == 3 ? ordinary_allocation() : malloc(64));
    if (kind != 2)
      free(const_cast<int *>(p));
    p[kind == 2 ? 16 : 0] = 1;
  }
}

// UAF: ERROR: AddressSanitizer: heap-use-after-free
// OVERFLOW: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK: Allocation token: 0x0
// CHECK: Token in report buffer
// PLAIN: ERROR: AddressSanitizer: heap-use-after-free
// PLAIN-NOT: Allocation token:
// REUSE: Allocation token: 0x0
// REUSE: 40-byte region
// REUSE-NOT: Allocation token:
