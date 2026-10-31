// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-ogcg.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-ogcg.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -fwrapv
// RUN: FileCheck --check-prefix=CIR_NO_POISON --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -fwrapv
// RUN: FileCheck --check-prefix=LLVM_NO_POISON --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-ogcg-wrapv.ll -fwrapv
// RUN: FileCheck --check-prefix=LLVM_NO_POISON --input-file=%t-ogcg-wrapv.ll %s

void *memcpy(void *, const void *, unsigned long);
void testMemcpy(void *dst, const void *src, unsigned long size) {
  memcpy(dst, src, size);
  // CHECK: cir.libc.memcpy
}

void *mempcpy(void *, const void *, unsigned long);
void *testMempcpy(void *dst, const void *src, unsigned long size) {
  return mempcpy(dst, src, size);
  // CHECK: cir.libc.memcpy
  // CHECK: cir.ptr_stride
}

void *memmove(void *, const void *, unsigned long);
void testMemmove(void *src, const void *dst, unsigned long size) {
  memmove(dst, src, size);
  // CHECK: cir.libc.memmove %{{.+}} bytes from %{{.+}} to %{{.+}} : !cir.ptr<!void>, !u64i
  // LLVM: call void @llvm.memmove.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 {{.*}}, i1 false)
}

void *memset(void *, int, unsigned long);
void testMemset(void *dst, int val, unsigned long size) {
  memset(dst, val, size);
  // CHECK: cir.libc.memset %{{.+}} bytes at %{{.+}} to %{{.+}} : !cir.ptr<!void>, !u8i, !u64i
  // LLVM: call void @llvm.memset.p0.i64(ptr {{.*}}, i8 {{.*}}, i64 {{.*}}, i1 false)
}

void *testBuiltinMemmove(void *dst, const void *src, unsigned long size) {
  return __builtin_memmove(dst, src, size);
  // CHECK-LABEL: cir.func {{.*}} @testBuiltinMemmove(
  // CHECK: %[[DST:.+]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CHECK: cir.libc.memmove %{{.+}} bytes from %{{.+}} align(1) to %[[DST]] align(1) : !cir.ptr<!void>, !u64i
  // CHECK: cir.store %[[DST]], %{{.+}} : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // LLVM-LABEL: define {{.*}} ptr @testBuiltinMemmove(
  // LLVM: call void @llvm.memmove.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 {{.*}}, i1 false)
}

void *testBuiltinMemset(void *dst, int val, unsigned long size) {
  return __builtin_memset(dst, val, size);
  // CHECK-LABEL: cir.func {{.*}} @testBuiltinMemset(
  // CHECK: %[[DST:.+]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CHECK: %[[VAL:.+]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
  // CHECK: %[[BYTE:.+]] = cir.cast integral %[[VAL]] : !s32i -> !u8i
  // CHECK: cir.libc.memset %{{.+}} bytes at %[[DST]] align(1) to %[[BYTE]] : !cir.ptr<!void>, !u8i, !u64i
  // CHECK: cir.store %[[DST]], %{{.+}} : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // LLVM-LABEL: define {{.*}} ptr @testBuiltinMemset(
  // LLVM: [[VAL:%.*]] = load i32, ptr
  // LLVM: [[BYTE:%.*]] = trunc i32 [[VAL]] to i8
  // LLVM: call void @llvm.memset.p0.i64(ptr {{.*}}, i8 [[BYTE]], i64 {{.*}}, i1 false)
}

// The alignment comes from the pointee type of the argument before its
// conversion to void *.
void testBuiltinAligned(int *dst, const int *src, unsigned long size) {
  __builtin_memmove(dst, src, size);
  __builtin_memset(dst, 0, size);
  // CHECK-LABEL: cir.func {{.*}} @testBuiltinAligned(
  // CHECK: cir.libc.memmove %{{.+}} bytes from %{{.+}} align(4) to %{{.+}} align(4) : !cir.ptr<!void>, !u64i
  // CHECK: cir.libc.memset %{{.+}} bytes at %{{.+}} align(4) to %{{.+}} : !cir.ptr<!void>, !u8i, !u64i
  // LLVM-LABEL: define {{.*}} void @testBuiltinAligned(
  // LLVM: call void @llvm.memmove.p0.p0.i64(ptr align 4 %{{.+}}, ptr align 4 %{{.+}}, i64 %{{.+}}, i1 false)
  // LLVM: call void @llvm.memset.p0.i64(ptr align 4 %{{.+}}, i8 0, i64 %{{.+}}, i1 false)
}

double fabs(double);
double testFabs(double x) {
  return fabs(x);
  // CHECK: cir.fabs %{{.+}} : !cir.double
}

float fabsf(float);
float testFabsf(float x) {
  return fabsf(x);
  // CHECK: cir.fabs %{{.+}} : !cir.float
}

int abs(int);
int testAbs(int x) {
  return abs(x);
  // CHECK: cir.abs %{{.+}} min_is_poison : !s32i
  // LLVM: %{{.+}} = call i32 @llvm.abs.i32(i32 %{{.+}}, i1 true)
  // CIR_NO_POISON: cir.abs %{{[^ ]+}} : !s32i
  // LLVM_NO_POISON: %{{.+}} = call i32 @llvm.abs.i32(i32 %{{.+}}, i1 false)
}

long labs(long);
long testLabs(long x) {
  return labs(x);
  // CHECK: cir.abs %{{.+}} min_is_poison : !s64i
  // LLVM: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 true)
  // CIR_NO_POISON: cir.abs %{{[^ ]+}} : !s64i
  // LLVM_NO_POISON: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 false)
}

long long llabs(long long);
long long testLlabs(long long x) {
  return llabs(x);
  // CHECK: cir.abs %{{.+}} min_is_poison : !s64i
  // LLVM: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 true)
  // CIR_NO_POISON: cir.abs %{{[^ ]+}} : !s64i
  // LLVM_NO_POISON: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 false)
}
