// RUN: %clang_cc1 -mprefer-vector-width=128 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK128
// RUN: %clang_cc1 -mprefer-vector-width=256 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK256
// RUN: %clang_cc1 -mprefer-vector-width=none -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECKNONE

int baz(int a) { return 4; }

// CHECK128: baz{{.*}} #0
// CHECK128: #0 = {{.*}}"prefer-vector-width"="128"

// CHECK256: baz{{.*}} #0
// CHECK256: #0 = {{.*}}"prefer-vector-width"="256"

// CHECKNONE: baz{{.*}} #0
// CHECKNONE-NOT: #0 = {{.*}}"prefer-vector-width"="none"
