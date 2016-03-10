// RUN: %clang_cl_asan -O0 %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <malloc.h>

int main() {
  while (true) {
    void *ptr = malloc(200 * 1024 * 1024);  // 200MB
  }
// CHECK: failed to allocate
}
