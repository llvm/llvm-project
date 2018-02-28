// RUN: %clang_cc1 %s -ast-print -fms-extensions | FileCheck %s

// CHECK: int x __attribute__((aligned(4)));
int x __attribute__((aligned(4)));

// FIXME: Print this at a valid location for a __declspec attr.
// CHECK: int y __declspec(align(4));
__declspec(align(4)) int y;

// CHECK: short arr[3] __attribute__((aligned));
short arr[3] __attribute__((aligned));

// CHECK: void foo() __attribute__((const));
void foo() __attribute__((const));

// CHECK: void bar() __attribute__((__const));
void bar() __attribute__((__const));

// CHECK: int * __ptr32 p32;
int * __ptr32 p32;

// CHECK: int * __ptr64 p64;
int * __ptr64 p64;

// TODO: the Type Printer has no way to specify the order to print attributes
// in, and so it currently always prints them in reverse order. Fix this.
// CHECK: int * __ptr32 __uptr p32_2;
int * __uptr __ptr32 p32_2;

// CHECK: int * __ptr64 __sptr p64_2;
int * __sptr __ptr64 p64_2;

// CHECK: int * __ptr32 __uptr p32_3;
int * __uptr __ptr32 p32_3;

// CHECK: int * __sptr * __ptr32 ppsp32;
int * __sptr * __ptr32 ppsp32;

// CHECK: __attribute__((availability(macos, strict, introduced=10.6)));
void f6(int) __attribute__((availability(macosx,strict,introduced=10.6)));
