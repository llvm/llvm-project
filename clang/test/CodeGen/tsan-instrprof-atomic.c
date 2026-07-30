// RUN: %clang_cc1 %s -emit-llvm -fprofile-instrument=clang -fprofile-update=atomic -o - | FileCheck %s
// RUN: %clang %s -S -emit-llvm -fprofile-generate -fprofile-update=atomic -o - | FileCheck %s
// RUN: %clang -O3 %s -S -emit-llvm -fprofile-generate -fprofile-update=atomic -o - | FileCheck %s

// CHECK: define {{.*}}@foo
// CHECK-NOT: load {{.*}}foo
// CHECK: ret void
void foo(void) {}
