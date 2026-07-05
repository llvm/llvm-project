// RUN: %clangxx %s -o %t && %t | FileCheck %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=covered && %t | FileCheck -check-prefix=CHECK-C %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=atomics && %t | FileCheck -check-prefix=CHECK-A %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=uar && %t | FileCheck -check-prefix=CHECK-U %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=covered,atomics && %t | FileCheck -check-prefix=CHECK-CA %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=covered,uar && %t | FileCheck -check-prefix=CHECK-CU %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=atomics,uar && %t | FileCheck -check-prefix=CHECK-AU %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=covered,atomics,uar && %t | FileCheck -check-prefix=CHECK-CAU %s
// RUN: %clangxx %s -o %t -mcmodel=large -fexperimental-sanitize-metadata=covered,atomics,uar && %t | FileCheck -check-prefix=CHECK-CAU %s

const int const_global = 42;

__attribute__((noinline, not_tail_called)) void escape(const volatile void *p) {
  [[maybe_unused]] static const volatile void *sink;
  sink = p;
}

// CHECK-NOT: metadata add
// CHECK: main
// CHECK-NOT: metadata del

// CHECK-C:      empty: features=0
// CHECK-A-NOT:  empty:
// CHECK-U-NOT:  empty:
// CHECK-CA:     empty: features=0
// CHECK-CU:     empty: features=0
// CHECK-AU-NOT: empty:
// CHECK-CAU:    empty: features=0
void empty() {}

// CHECK-C:  normal: features=0
// CHECK-A:  normal: features=1
// CHECK-U:  normal: features=2
// CHECK-CA: normal: features=1
// CHECK-CU: normal: features=2
// CHECK-AU: normal: features=3
// CHECK-CAU:normal: features=3
void normal() {
  int x;
  escape(&x);
}

// CHECK-C:      with_const_global: features=0
// CHECK-A-NOT:  with_const_global:
// CHECK-U-NOT:  with_const_global:
// CHECK-CA:     with_const_global: features=0
// CHECK-CU:     with_const_global: features=0
// CHECK-AU-NOT: with_const_global:
// CHECK-CAU:    with_const_global: features=0
int with_const_global() { return *((const volatile int *)&const_global); }

// CHECK-C:     with_atomic: features=0
// CHECK-A:     with_atomic: features=1
// CHECK-U-NOT: with_atomic:
// CHECK-CA:    with_atomic: features=1
// CHECK-CU:    with_atomic: features=0
// CHECK-AU:    with_atomic: features=1
// CHECK-CAU:   with_atomic: features=1
int with_atomic(int *p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }

// CHECK-C:   with_atomic_escape: features=0
// CHECK-A:   with_atomic_escape: features=1
// CHECK-U:   with_atomic_escape: features=2
// CHECK-CA:  with_atomic_escape: features=1
// CHECK-CU:  with_atomic_escape: features=2
// CHECK-AU:  with_atomic_escape: features=3
// CHECK-CAU: with_atomic_escape: features=3
int with_atomic_escape(int *p) {
  escape(&p);
  return __atomic_load_n(p, __ATOMIC_RELAXED);
}

// CHECK-C:   with_atomic_escape_lots_of_args: features=0
// CHECK-A:   with_atomic_escape_lots_of_args: features=1
// CHECK-U:   with_atomic_escape_lots_of_args: features=6
// CHECK-CA:  with_atomic_escape_lots_of_args: features=1
// CHECK-CU:  with_atomic_escape_lots_of_args: features=6
// CHECK-AU:  with_atomic_escape_lots_of_args: features=7
// CHECK-CAU: with_atomic_escape_lots_of_args: features=7
long with_atomic_escape_lots_of_args(int *p, long a0, long a1, long a2, long a3,
                                     long a4, long a5, long a6) {
  escape(&p);
  return a0 + a1 + a2 + a3 + a4 + a5 + a6 +
         __atomic_load_n(p, __ATOMIC_RELAXED);
}

// CHECK-C:     ellipsis: features=0
// CHECK-A:     ellipsis: features=1
// CHECK-U-NOT: ellipsis:
// CHECK-CA:    ellipsis: features=1
// CHECK-CU:    ellipsis: features=0
// CHECK-AU:    ellipsis: features=1
// CHECK-CAU:   ellipsis: features=1
void ellipsis(int *p, ...) {
  escape(&p);
  [[maybe_unused]] volatile int x;
  x = 0;
}

// CHECK-C:     ellipsis_with_atomic: features=0
// CHECK-A:     ellipsis_with_atomic: features=1
// CHECK-U-NOT: ellipsis_with_atomic:
// CHECK-CA:    ellipsis_with_atomic: features=1
// CHECK-CU:    ellipsis_with_atomic: features=0
// CHECK-AU:    ellipsis_with_atomic: features=1
// CHECK-CAU:   ellipsis_with_atomic: features=1
int ellipsis_with_atomic(int *p, ...) {
  escape(&p);
  return __atomic_load_n(p, __ATOMIC_RELAXED);
}

#define FUNCTIONS                                                              \
  FN(empty);                                                                   \
  FN(normal);                                                                  \
  FN(with_const_global);                                                       \
  FN(with_atomic);                                                             \
  FN(with_atomic_escape);                                                      \
  FN(with_atomic_escape_lots_of_args);                                         \
  FN(ellipsis);                                                                \
  FN(ellipsis_with_atomic);                                                    \
  /**/

#include "common.h"
