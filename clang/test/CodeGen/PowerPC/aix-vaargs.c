// REQUIRES: powerpc-registered-target
// REQUIRES: asserts
// RUN: %clang_cc1 -triple powerpc-unknown-aix -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AIX32
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AIX64

struct x {
  double b;
  long a;
};

void testva (int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  struct x t = __builtin_va_arg(ap, struct x);
  __builtin_va_list ap2;
  __builtin_va_copy(ap2, ap);
  int v = __builtin_va_arg(ap2, int);
  __builtin_va_end(ap2);
  __builtin_va_end(ap);
}

// AIX32: define void @testva(i32 noundef %n, ...)
// AIX64: define void @testva(i32 noundef signext %n, ...)

// CHECK-NEXT: entry:
// CHECK-NEXT:  %n.addr = alloca i32, align 4

// AIX32-NEXT:  %ap = alloca ptr, align 4
// AIX64-NEXT:  %ap = alloca ptr, align 8

// CHECK-NEXT:  %t = alloca %struct.x, align 8

// AIX32-NEXT:  %ap2 = alloca ptr, align 4
// AIX64-NEXT:  %ap2 = alloca ptr, align 8

// CHECK-NEXT:  %v = alloca i32, align 4
// CHECK-NEXT:  store i32 %n, ptr %n.addr, align 4
// CHECK-NEXT:  call void @llvm.va_start(ptr %ap)

// AIX32-NEXT:  %argp.cur = load ptr, ptr %ap, align 4
// AIX32-NEXT:  %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 16
// AIX32-NEXT:  store ptr %argp.next, ptr %ap, align 4
// AIX64-NEXT:  %argp.cur = load ptr, ptr %ap, align 8
// AIX64-NEXT:  %argp.next = getelementptr inbounds i8, ptr %argp.cur, i64 16
// AIX64-NEXT:  store ptr %argp.next, ptr %ap, align 8


// AIX32-NEXT:  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %t, ptr align 4 %argp.cur, i32 16, i1 false)
// AIX64-NEXT:  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %t, ptr align 8 %argp.cur, i64 16, i1 false)

// CHECK-NEXT:  call void @llvm.va_copy(ptr %ap2, ptr %ap)

// AIX32-NEXT:  %argp.cur1 = load ptr, ptr %ap2, align 4
// AIX32-NEXT:  %argp.next2 = getelementptr inbounds i8, ptr %argp.cur1, i32 4
// AIX32-NEXT:  store ptr %argp.next2, ptr %ap2, align 4
// AIX32-NEXT:  %0 = load i32, ptr %argp.cur1, align 4
// AIX32-NEXT:  store i32 %0, ptr %v, align 4
// AIX64-NEXT:  %argp.cur1 = load ptr, ptr %ap2, align 8
// AIX64-NEXT:  %argp.next2 = getelementptr inbounds i8, ptr %argp.cur1, i64 8
// AIX64-NEXT:  store ptr %argp.next2, ptr %ap2, align 8
// AIX64-NEXT:  %0 = getelementptr inbounds i8, ptr %argp.cur1, i64 4
// AIX64-NEXT:  %1 = load i32, ptr %0, align 4
// AIX64-NEXT:  store i32 %1, ptr %v, align 4

// CHECK-NEXT:  call void @llvm.va_end(ptr %ap2)
// CHECK-NEXT:  call void @llvm.va_end(ptr %ap)
// CHECK-NEXT:  ret void

// CHECK: declare void @llvm.va_start(ptr)

// AIX32: declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
// AIX64: declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

// CHECK: declare void @llvm.va_copy(ptr, ptr)
// CHECK: declare void @llvm.va_end(ptr)
