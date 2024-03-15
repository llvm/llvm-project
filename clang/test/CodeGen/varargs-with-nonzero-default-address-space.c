// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fcuda-is-device -emit-llvm -o - %s | FileCheck %s
struct x {
  double b;
  long a;
};

void testva(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  struct x t = __builtin_va_arg(ap, struct x);
  __builtin_va_list ap2;
  __builtin_va_copy(ap2, ap);
  int v = __builtin_va_arg(ap2, int);
  __builtin_va_end(ap2);
  __builtin_va_end(ap);
}

// CHECK:  call void @llvm.va_start.p4(ptr addrspace(4) %ap{{.*}})
// CHECK:  call void @llvm.va_copy.p4.p4(ptr addrspace(4) %ap2{{.*}}, ptr addrspace(4) {{.*}})
// CHECK:  call void @llvm.va_end.p4(ptr addrspace(4) %ap2{{.*}})
// CHECK-NEXT:  call void @llvm.va_end.p4(ptr addrspace(4) %ap{{.*}})