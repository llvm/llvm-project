// RUN: %clangxx %s -o %t && %t | FileCheck %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=covered && %t | FileCheck -check-prefix=CHECK-C %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=atomics && %t | FileCheck -check-prefix=CHECK-A %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=uar && %t | FileCheck -check-prefix=CHECK-U %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=covered,atomics && %t | FileCheck -check-prefix=CHECK-CA %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=covered,uar && %t | FileCheck -check-prefix=CHECK-CU %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=atomics,uar && %t | FileCheck -check-prefix=CHECK-AU %s
// RUN: %clangxx %s -o %t -fexperimental-sanitize-metadata=covered,atomics,uar && %t | FileCheck -check-prefix=CHECK-CAU %s

// CHECK-NOT: metadata add
// CHECK: main
// CHECK-NOT: metadata del

// CHECK-C:      empty: features=0
// CHECK-A-NOT:  empty:
// CHECK-U-NOT:  empty:
// CHECK-CA:     empty: features=1
// CHECK-CU:     empty: features=0
// CHECK-AU-NOT: empty:
// CHECK-CAU:    empty: features=1
void empty() {}

// CHECK-C:  normal: features=0
// CHECK-A:  normal: features=1
// CHECK-U:  normal: features=2
// CHECK-CA: normal: features=1
// CHECK-CU: normal: features=2
// CHECK-AU: normal: features=3
// CHECK-CAU:normal: features=3
void normal() {
  volatile int x;
  x = 0;
}

// CHECK-C:   with_atomic: features=0
// CHECK-A:   with_atomic: features=1
// CHECK-U:   with_atomic: features=2
// CHECK-CA:  with_atomic: features=1
// CHECK-CU:  with_atomic: features=2
// CHECK-AU:  with_atomic: features=3
// CHECK-CAU: with_atomic: features=3
int with_atomic(int *p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }

// CHECK-C:     ellipsis: features=0
// CHECK-A:     ellipsis: features=1
// CHECK-U-NOT: ellipsis:
// CHECK-CA:    ellipsis: features=1
// CHECK-CU:    ellipsis: features=0
// CHECK-AU:    ellipsis: features=1
// CHECK-CAU:   ellipsis: features=1
void ellipsis(int *p, ...) {
  volatile int x;
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
  return __atomic_load_n(p, __ATOMIC_RELAXED);
}

#define FUNCTIONS                                                              \
  FN(empty);                                                                   \
  FN(normal);                                                                  \
  FN(with_atomic);                                                             \
  FN(ellipsis);                                                                \
  FN(ellipsis_with_atomic);                                                    \
  /**/

#include "common.h"
