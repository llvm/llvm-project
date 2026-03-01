// RUN: %clang_cc1 -emit-llvm -O0 -cl-std=CL1.2 -o - %s | FileCheck %s --check-prefix=CHECK-UNIFORM
// RUN: %clang_cc1 -emit-llvm -O0 -cl-std=CL2.0 -o - %s | FileCheck %s --check-prefix=CHECK-NONUNIFORM
// RUN: %clang_cc1 -emit-llvm -O0 -cl-std=CL2.0 -cl-uniform-work-group-size -o - %s | FileCheck %s --check-prefix=CHECK-UNIFORM
// RUN: %clang_cc1 -emit-llvm -O0 -cl-std=CL2.0 -foffload-uniform-block -o - %s | FileCheck %s --check-prefix=CHECK-UNIFORM

// CHECK-UNIFORM: define dso_local spir_kernel void @ker(){{.*}}[[ATTR0:#[0-9]+]]
// CHECK-UNIFORM: define dso_local void @__clang_ocl_kern_imp_ker(){{.*}}[[ATTR1:#[0-9]+]]
// CHECK-UNIFORM: define dso_local void @foo{{.*}}[[ATTR1]]

// CHECK-NONUNIFORM: define dso_local spir_kernel void @ker(){{.*}}[[ATTR0:#[0-9]+]]
// CHECK-NONUNIFORM: define dso_local void @__clang_ocl_kern_imp_ker(){{.*}}[[ATTR0]]
// CHECK-NONUNIFORM: define dso_local void @foo{{.*}}[[ATTR0]]
kernel void ker() {};

void foo() {};

// CHECK-UNIFORM: attributes [[ATTR0]] {{.*}} "uniform-work-group-size"
// CHECK-UNIFORM-NOT: attributes [[ATTR1]] {{.*}} "uniform-work-group-size"

// CHECK-NONUNIFORM-NOT: attributes [[ATTR0]] {{.*}} "uniform-work-group-size"
