// RUN: %clang_cc1 -triple i686-w64-mingw32 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
//
// Verifies that clang detects memcpy inline version and uses it instead of the builtin.
// Checks that clang correctly walks through multiple forward declaration.

typedef unsigned int size_t;

void *memcpy(void *_Dst, const void *_Src, size_t _Size);

extern __inline__ __attribute__((__always_inline__, __gnu_inline__)) __attribute__((__artificial__))
void *memcpy(void *__dst, const void *__src, size_t __n)
{
  return __builtin___memcpy_chk(__dst, __src, __n, __builtin_object_size((__dst), ((0) > 0) && (2 > 1)));
}

void *memcpy(void *_Dst, const void *_Src, size_t _Size);

char *a, *b;
void func(void) {
    memcpy(a, b, 42);
}

// CHECK-LABEL: define {{.*}} @func(
// CHECK: call ptr @memcpy.inline

// CHECK-LABEL: declare {{.*}} @memcpy(

// CHECK-LABEL: define {{.*}} @memcpy.inline(
// CHECK: call ptr @__memcpy_chk
