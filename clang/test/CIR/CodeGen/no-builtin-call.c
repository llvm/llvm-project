// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - \
// RUN:   | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=LLVM %s

// __attribute__((no_builtin)) on the calling function turns calls to the named
// library functions into ordinary calls.

typedef __SIZE_TYPE__ size_t;

void *memset(void *s, int c, size_t n);
void *memcpy(void *d, const void *s, size_t n);
void *memmove(void *d, const void *s, size_t n);

void none(char *s, char *d, size_t n) {
  memset(s, 0, n);
  memcpy(d, s, n);
  memmove(d, s, n);
}

// CIR-LABEL: cir.func {{.*}} @none(
// CIR:         cir.libc.memset
// CIR:         cir.libc.memcpy
// CIR:         cir.libc.memmove

// LLVM-LABEL: define {{.*}} void @none(
// LLVM:         call void @llvm.memset.p0.i64(
// LLVM:         call void @llvm.memcpy.p0.p0.i64(
// LLVM:         call void @llvm.memmove.p0.p0.i64(

__attribute__((no_builtin("memset"))) void no_memset(char *s, char *d,
                                                     size_t n) {
  memset(s, 0, n);
  memcpy(d, s, n);
  memmove(d, s, n);
}

// CIR-LABEL: cir.func {{.*}} @no_memset(
// CIR:         cir.call @memset(
// CIR:         cir.libc.memcpy
// CIR:         cir.libc.memmove

// LLVM-LABEL: define {{.*}} void @no_memset(
// LLVM:         call ptr @memset(
// LLVM:         call void @llvm.memcpy.p0.p0.i64(
// LLVM:         call void @llvm.memmove.p0.p0.i64(

__attribute__((no_builtin("memcpy", "memmove"))) void
no_memcpy_memmove(char *s, char *d, size_t n) {
  memset(s, 0, n);
  memcpy(d, s, n);
  memmove(d, s, n);
}

// CIR-LABEL: cir.func {{.*}} @no_memcpy_memmove(
// CIR:         cir.libc.memset
// CIR:         cir.call @memcpy(
// CIR:         cir.call @memmove(

// LLVM-LABEL: define {{.*}} void @no_memcpy_memmove(
// LLVM:         call void @llvm.memset.p0.i64(
// LLVM:         call ptr @memcpy(
// LLVM:         call ptr @memmove(

__attribute__((no_builtin)) void no_builtins(char *s, char *d, size_t n) {
  memset(s, 0, n);
  memcpy(d, s, n);
  memmove(d, s, n);
}

// CIR-LABEL: cir.func {{.*}} @no_builtins(
// CIR:         cir.call @memset(
// CIR:         cir.call @memcpy(
// CIR:         cir.call @memmove(

// LLVM-LABEL: define {{.*}} void @no_builtins(
// LLVM:         call ptr @memset(
// LLVM:         call ptr @memcpy(
// LLVM:         call ptr @memmove(

// The attribute only applies to library functions, not to __builtin_ names.
__attribute__((no_builtin)) void builtin_names(char *s, char *d, size_t n) {
  __builtin_memset(s, 0, n);
  __builtin_memcpy(d, s, n);
  __builtin_memmove(d, s, n);
}

// CIR-LABEL: cir.func {{.*}} @builtin_names(
// CIR:         cir.libc.memset
// CIR:         cir.libc.memcpy
// CIR:         cir.libc.memmove

// LLVM-LABEL: define {{.*}} void @builtin_names(
// LLVM:         call void @llvm.memset.p0.i64(
// LLVM:         call void @llvm.memcpy.p0.p0.i64(
// LLVM:         call void @llvm.memmove.p0.p0.i64(
