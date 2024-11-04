// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -fwrapv
// RUN: FileCheck --check-prefix=CIR_NO_POISON --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -fwrapv
// RUN: FileCheck --check-prefix=LLVM_NO_POISON --input-file=%t.ll %s

// Should generate CIR's builtin memcpy op.
void *memcpy(void *, const void *, unsigned long);
void testMemcpy(void *dst, const void *src, unsigned long size) {
  memcpy(dst, src, size);
  // CHECK: cir.libc.memcpy %{{.+}} bytes from %{{.+}} to %{{.+}} : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>
}

// Should generate CIR's builtin memmove op.
void *memmove(void *, const void *, unsigned long);
void testMemmove(void *src, const void *dst, unsigned long size) {
  memmove(dst, src, size);
  // CHECK: cir.libc.memmove %{{.+}} bytes from %{{.+}} to %{{.+}} : !cir.ptr<!void>, !u64i
  // LLVM: call void @llvm.memmove.{{.+}}.i64(ptr %{{.+}}, ptr %{{.+}}, i64 %{{.+}}, i1 false),
}

// Should generate CIR's builtin memset op.
void *memset(void *, int, unsigned long);
void testMemset(void *dst, int val, unsigned long size) {
  memset(dst, val, size);
  // CHECK: cir.libc.memset %{{.+}} bytes from %{{.+}} set to %{{.+}} : !cir.ptr<!void>, !s32i, !u64i
  // LLVM: call void @llvm.memset.{{.+}}.i64(ptr %{{.+}}, i8 %{{.+}}, i64 %{{.+}}, i1 false)
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
  // CHECK: cir.abs %{{.+}} poison : !s32i
  // LLVM: %{{.+}} = call i32 @llvm.abs.i32(i32 %{{.+}}, i1 true)
  // CIR_NO_POISON: cir.abs %{{.+}} : !s32i
  // LLVM_NO_POISON: %{{.+}} = call i32 @llvm.abs.i32(i32 %{{.+}}, i1 false)
}

long labs(long);
long testLabs(long x) {
  return labs(x);
  // CHECK: cir.abs %{{.+}} poison : !s64i
  // LLVM: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 true)
  // CIR_NO_POISON: cir.abs %{{.+}} : !s64i
  // LLVM_NO_POISON: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 false)
}

long long llabs(long long);
long long testLlabs(long long x) {
  return llabs(x);
  // CHECK: cir.abs %{{.+}} poison : !s64i
  // LLVM: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 true)
  // CIR_NO_POISON: cir.abs %{{.+}} : !s64i
  // LLVM_NO_POISON: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 false)
}
