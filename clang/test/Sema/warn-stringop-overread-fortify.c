// Check that _FORTIFY_SOURCE can interoperate with -Wstringop-overread without
// interfering with one another. We model it here like glibc does so this test
// is target independent.
//
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -D_FORTIFY_SOURCE=1 %s -verify
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -D_FORTIFY_SOURCE=2 %s -verify
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -D_FORTIFY_SOURCE=3 %s -verify
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -D_FORTIFY_SOURCE=2 -fsanitize=address %s -verify
//
// Confirm we actually see _chk wrappers:
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -D_FORTIFY_SOURCE=2 %s -o - | FileCheck %s

typedef __SIZE_TYPE__ size_t;

#if defined(_FORTIFY_SOURCE) && _FORTIFY_SOURCE > 0
extern __inline __attribute__((__always_inline__, __gnu_inline__))
void *memcpy(void *dst, const void *src, size_t len) {
  return __builtin___memcpy_chk(dst, src, len, __builtin_object_size(dst, 0));
}
#else
void *memcpy(void *dst, const void *src, size_t c);
#endif

void test(void) {
  char dst[100];
  char src[4];
  // CHECK: call {{.*}}@__memcpy_chk
  memcpy(dst, src, 8);  // expected-warning {{'memcpy' reading 8 bytes from a region of size 4}}
}
