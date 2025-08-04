// RUN: %clang_cc1 -emit-llvm -O0 -cl-std=CL1.2 -o - %s 2>&1 | FileCheck %s -check-prefixes CHECK,CHECK-UNIFORM
// RUN: %clang_cc1 -emit-llvm -O0 -cl-std=CL2.0 -o - %s 2>&1 | FileCheck %s -check-prefixes CHECK,CHECK-NONUNIFORM
// RUN: %clang_cc1 -emit-llvm -O0 -cl-std=CL2.0 -cl-uniform-work-group-size -o - %s 2>&1 | FileCheck %s -check-prefixes CHECK,CHECK-UNIFORM
// RUN: %clang_cc1 -emit-llvm -O0 -cl-std=CL2.0 -foffload-uniform-block -o - %s 2>&1 | FileCheck %s -check-prefixes CHECK,CHECK-UNIFORM

kernel void ker() {};
// CHECK: define{{.*}}@ker() #[[ATTR0:[0-9]+]]

// CHECK: define{{.*}}@__clang_ocl_kern_imp_ker() #[[ATTR1:[0-9]+]]

void foo() {};
// CHECK: define{{.*}}@foo() #[[ATTR1:[0-9]+]]

// CHECK: attributes #[[ATTR0]]
// CHECK-UNIFORM: "uniform-work-group-size"="true"
// CHECK-NONUNIFORM: "uniform-work-group-size"="false"

// CHECK: attributes #[[ATTR1]]
// CHECK-NOT: uniform-work-group-size
