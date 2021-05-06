// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm < %s | FileCheck %s --check-prefix=ALIGN16
// RUN: %clang_cc1 -triple x86_64-windows-msvc      -emit-llvm < %s | FileCheck %s --check-prefix=ALIGN16
// RUN: %clang_cc1 -triple i386-apple-darwin        -emit-llvm < %s | FileCheck %s --check-prefix=ALIGN16
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu   -emit-llvm < %s | FileCheck %s --check-prefix=ALIGN8
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-malloc  -emit-llvm < %s  | FileCheck %s --check-prefix=NOBUILTIN-MALLOC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-calloc  -emit-llvm < %s  | FileCheck %s --check-prefix=NOBUILTIN-CALLOC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-realloc -emit-llvm < %s  | FileCheck %s --check-prefix=NOBUILTIN-REALLOC

typedef __SIZE_TYPE__ size_t;

void *malloc(size_t);
void *calloc(size_t, size_t);
void *realloc(void *, size_t);

void *malloc_test(size_t n) {
  return malloc(n);
}

void *calloc_test(size_t n) {
  return calloc(1, n);
}

void *raalloc_test(void *p, size_t n) {
  return realloc(p, n);
}

// ALIGN16: align 16 i8* @malloc
// ALIGN16: align 16 i8* @calloc
// ALIGN16: align 16 i8* @realloc
// ALIGN8: align 8 i8* @malloc
// ALIGN8: align 8 i8* @calloc
// ALIGN8: align 8 i8* @realloc
// NOBUILTIN-MALLOC: declare i8* @malloc
// NOBUILTIN-CALLOC: declare i8* @calloc
// NOBUILTIN-REALLOC: declare i8* @realloc
