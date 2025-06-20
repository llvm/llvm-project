// REQUIRES: x86-registered-target

// Make sure opt-bisect works through both pass managers
//
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O1 %s -mllvm -opt-disable="inlinerpass,SROAPass,machine code sinking" -mllvm -opt-disable-verbose -emit-obj -o /dev/null 2>&1 | FileCheck %s

// CHECK-NOT: DISABLE: running pass InlinerPass
// CHECK-NOT: DISABLE: running pass SROAPass
// CHECK-NOT: DISABLE: running pass Machine code sinking
// Make sure that legacy pass manager is running
// CHECK: Instruction Selection

int func(int a) { return a; }
