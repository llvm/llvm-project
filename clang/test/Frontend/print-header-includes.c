// RUN: %clang_cc1 -include Inputs/test3.h -E -H -o %t.out %s 2> %t.err
// RUN: FileCheck < %t.err %s

// CHECK-NOT: test3.h
// CHECK: . {{.*test.h}}
// CHECK: .. {{.*test2.h}}

#include "Inputs/test.h"
