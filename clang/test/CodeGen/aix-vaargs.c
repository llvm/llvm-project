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

// AIX32: define void @testva(i32 %n, ...)
// AIX64: define void @testva(i32 signext %n, ...)

// CHECK-NEXT: entry:
// CHECK-NEXT:  %n.addr = alloca i32, align 4

// AIX32-NEXT:  %ap = alloca i8*, align 4
// AIX64-NEXT:  %ap = alloca i8*, align 8

// CHECK-NEXT:  %t = alloca %struct.x, align 8

// AIX32-NEXT:  %ap2 = alloca i8*, align 4
// AIX64-NEXT:  %ap2 = alloca i8*, align 8

// CHECK-NEXT:  %v = alloca i32, align 4
// CHECK-NEXT:  store i32 %n, i32* %n.addr, align 4
// CHECK-NEXT:  %ap1 = bitcast i8** %ap to i8*
// CHECK-NEXT:  call void @llvm.va_start(i8* %ap1)

// AIX32-NEXT:  %argp.cur = load i8*, i8** %ap, align 4
// AIX32-NEXT:  %argp.next = getelementptr inbounds i8, i8* %argp.cur, i32 16
// AIX32-NEXT:  store i8* %argp.next, i8** %ap, align 4
// AIX64-NEXT:  %argp.cur = load i8*, i8** %ap, align 8
// AIX64-NEXT:  %argp.next = getelementptr inbounds i8, i8* %argp.cur, i64 16
// AIX64-NEXT:  store i8* %argp.next, i8** %ap, align 8

// CHECK-NEXT:  %0 = bitcast i8* %argp.cur to %struct.x*
// CHECK-NEXT:  %1 = bitcast %struct.x* %t to i8*
// CHECK-NEXT:  %2 = bitcast %struct.x* %0 to i8*

// AIX32-NEXT:  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %1, i8* align 4 %2, i32 16, i1 false)
// AIX64-NEXT:  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %1, i8* align 8 %2, i64 16, i1 false)

// CHECK-NEXT:  %3 = bitcast i8** %ap2 to i8*
// CHECK-NEXT:  %4 = bitcast i8** %ap to i8*
// CHECK-NEXT:  call void @llvm.va_copy(i8* %3, i8* %4)

// AIX32-NEXT:  %argp.cur2 = load i8*, i8** %ap2, align 4
// AIX32-NEXT:  %argp.next3 = getelementptr inbounds i8, i8* %argp.cur2, i32 4
// AIX32-NEXT:  store i8* %argp.next3, i8** %ap2, align 4
// AIX32-NEXT:  %5 = bitcast i8* %argp.cur2 to i32*
// AIX32-NEXT:  %6 = load i32, i32* %5, align 4
// AIX32-NEXT:  store i32 %6, i32* %v, align 4
// AIX64-NEXT:  %argp.cur2 = load i8*, i8** %ap2, align 8
// AIX64-NEXT:  %argp.next3 = getelementptr inbounds i8, i8* %argp.cur2, i64 8
// AIX64-NEXT:  store i8* %argp.next3, i8** %ap2, align 8
// AIX64-NEXT:  %5 = getelementptr inbounds i8, i8* %argp.cur2, i64 4
// AIX64-NEXT:  %6 = bitcast i8* %5 to i32*
// AIX64-NEXT:  %7 = load i32, i32* %6, align 4
// AIX64-NEXT:  store i32 %7, i32* %v, align 4

// CHECK-NEXT:  %ap24 = bitcast i8** %ap2 to i8*
// CHECK-NEXT:  call void @llvm.va_end(i8* %ap24)
// CHECK-NEXT:  %ap5 = bitcast i8** %ap to i8*
// CHECK-NEXT:  call void @llvm.va_end(i8* %ap5)
// CHECK-NEXT:  ret void

// CHECK: declare void @llvm.va_start(i8*)

// AIX32: declare void @llvm.memcpy.p0i8.p0i8.i32(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i32, i1 immarg)
// AIX64: declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)

// CHECK: declare void @llvm.va_copy(i8*, i8*)
// CHECK: declare void @llvm.va_end(i8*)
