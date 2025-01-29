// RUN: %clang_cc1 -triple=powerpc-ibm-aix-xcoff -emit-llvm < %s | \
// RUN:   FileCheck --check-prefix=32BIT %s

// RUN: %clang_cc1 -triple=powerpc64-ibm-aix-xcoff -emit-llvm < %s | \
// RUN:   FileCheck --check-prefix=64BIT %s

typedef __SIZE_TYPE__ size_t;
extern void *alloca(size_t __size) __attribute__((__nothrow__));

void foo(void) {
  char *ptr1 = (char *)alloca(sizeof(char) * 9);
  char *ptr2 = (char *)alloca(sizeof(char) * 32);
}

// 32BIT: %0 = alloca i8, i32 9, align 16
// 32BIT: %1 = alloca i8, i32 32, align 16

// 64BIT: %0 = alloca i8, i64 9, align 16
// 64BIT: %1 = alloca i8, i64 32, align 16
