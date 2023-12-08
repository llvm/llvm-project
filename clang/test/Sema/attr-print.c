// RUN: %clang_cc1 %s -ast-print -fms-extensions | FileCheck %s

// CHECK: int x __attribute__((aligned(4)));
int x __attribute__((aligned(4)));

// CHECK: __declspec(align(4)) int y;
__declspec(align(4)) int y;

// CHECK: short arr[3] __attribute__((aligned));
short arr[3] __attribute__((aligned));

// CHECK: void foo(void) __attribute__((const));
void foo(void) __attribute__((const));

// CHECK: void bar(void) __attribute__((__const));
void bar(void) __attribute__((__const));

// CHECK: int * __ptr32 p32;
int * __ptr32 p32;

// CHECK: int * __ptr64 p64;
int * __ptr64 p64;

// CHECK: int * __uptr __ptr32 p32_2;
int * __uptr __ptr32 p32_2;

// CHECK: int * __sptr __ptr64 p64_2;
int * __sptr __ptr64 p64_2;

// CHECK: int * __uptr __ptr32 p32_3;
int * __uptr __ptr32 p32_3;

// CHECK: int * __sptr * __ptr32 ppsp32;
int * __sptr * __ptr32 ppsp32;

// CHECK: __attribute__((availability(macos, strict, introduced=10.6)));
void f6(int) __attribute__((availability(macosx,strict,introduced=10.6)));
