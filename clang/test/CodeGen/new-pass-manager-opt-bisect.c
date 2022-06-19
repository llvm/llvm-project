// REQUIRES: x86-registered-target

// Make sure opt-bisect works through both pass managers
//
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O1 %s -mllvm -opt-bisect-limit=-1 -emit-obj -o /dev/null 2>&1 | FileCheck %s

// CHECK: BISECT: running pass (1)
// CHECK-NOT: BISECT: running pass (1)
// Make sure that legacy pass manager is running
// CHECK: Instruction Selection

int func(int a) { return a; }
