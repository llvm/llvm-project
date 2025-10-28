// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL2.0 -include opencl-c.h -triple spir64 -emit-llvm -o - | FileCheck %s

// CHECK: @constant_p_NULL =
// CHECK-SAME: addrspace(1) global ptr addrspace(2) null, align 8
constant char *constant_p_NULL = NULL;

// CHECK-LABEL: cmp_constant
// CHECK: icmp eq ptr addrspace(2) %p, null
char cmp_constant(constant char* p) {
  if (p != 0)
    return *p;
  else
    return 0;
}
