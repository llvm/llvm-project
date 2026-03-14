// RUN: mkdir -p %t.dir
// RUN: %clang_cc1 -E -fkeep-system-includes -I %S/Inputs/dashE -isystem %S/Inputs/dashE/sys %s | FileCheck %s

int main_1 = 1;
#include <a.h>
int main_2 = 1;
#include "dashE.h"
int main_3 = 1;

// CHECK: main_1
// CHECK: #include <a.h>
// CHECK-NOT: a_1
// CHECK-NOT: a_2
// CHECK-NOT: b.h
// CHECK: main_2
// CHECK-NOT: #include "dashE.h"
// CHECK: dashE_1
// CHECK: #include <a.h>
// CHECK-NOT: a_1
// CHECK-NOT: a_2
// CHECK-NOT: b.h
// CHECK: dashE_2
// CHECK: main_3
