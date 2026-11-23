// REQUIRES: nvptx-registered-target
// RUN: %clang -cc1 -triple nvptx64-nvidia-cuda -flto=thin -funified-lto -emit-llvm-bc %s -o %t.bc
// RUN: mkdir -p %t.dir
// RUN: cd %t.dir && clang-nvlink-wrapper --dry-run --save-temps --plugin-opt=thinlto -arch sm_90 %t.bc -o thin 2>&1 | FileCheck %s
// CHECK-NOT: ptxas{{.*}} -c  -O
// CHECK: ptxas{{.*}} -c thin.1.s {{.*}} -o thin.1.cubin
// CHECK-NEXT: {{.*}}nvlink{{.*}} -o thin thin.1.cubin

int foo(int x) { return x + 1; }
int bar(int x) { return x * 3; }
