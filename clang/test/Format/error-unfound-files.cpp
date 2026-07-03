// RUN: rm -f a.c b.c

// RUN: not clang-format a.c b.c 2>&1 | FileCheck %s
// CHECK: a.c:
// CHECK-NEXT: b.c:
