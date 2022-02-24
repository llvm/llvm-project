// RUN: %clang_cc1 -E -dM -triple=x86_64-none-none  %s -o - \
// RUN:   | FileCheck %s -strict-whitespace

// RUN: %clang_cc1 -E -dM -triple=x86_64-none-none -target-feature -sse \
// RUN:   %s -o - | FileCheck %s -check-prefix=EXT -strict-whitespace

// RUN: %clang_cc1 -E -dM -triple=arm64e-apple-ios -target-feature -sse \
// RUN:   %s -o - | FileCheck %s  -strict-whitespace

// RUN: %clang_cc1 -E -dM -triple=arm64e-apple-ios -target-feature +sse \
// RUN:   %s -o - | FileCheck %s  -strict-whitespace

// RUN: %clang_cc1 -E -dM -triple=arm64_32-apple-ios  %s -o - \
// RUN:   | FileCheck %s  -strict-whitespace

// RUN: %clang_cc1 -E -dM -triple=arm64_32-apple-ios -target-feature -sse \
// RUN:   %s -o - | FileCheck %s  -strict-whitespace

// RUN: %clang_cc1 -E -dM -triple i386-pc-windows -target-cpu pentium4 %s -o - \
// RUN:   | FileCheck %s  -strict-whitespace

// RUN: %clang_cc1 -E -dM -triple i386-pc-windows -target-cpu pentium4 \
// RUN:   -target-feature -sse %s -o - | FileCheck -check-prefix=EXT %s \
// RUN:   -strict-whitespace

#ifdef __FLT_EVAL_METHOD__
#if __FLT_EVAL_METHOD__ == 3
#define __GLIBC_FLT_EVAL_METHOD 2
#else
#define __GLIBC_FLT_EVAL_METHOD __FLT_EVAL_METHOD__
#endif
#elif defined __x86_64__
#define __GLIBC_FLT_EVAL_METHOD 0
#else
#define __GLIBC_FLT_EVAL_METHOD 2
#endif

#if __GLIBC_FLT_EVAL_METHOD == 0 || __GLIBC_FLT_EVAL_METHOD == 16
#define Name "One"
#elif __GLIBC_FLT_EVAL_METHOD == 1
#define Name "Two"
#elif __GLIBC_FLT_EVAL_METHOD == 2
#define Name "Three"
#elif __GLIBC_FLT_EVAL_METHOD == 32
#define Name "Four"
#elif __GLIBC_FLT_EVAL_METHOD == 33
#define Name "Five"
#elif __GLIBC_FLT_EVAL_METHOD == 64
#define Name "Six"
#elif __GLIBC_FLT_EVAL_METHOD == 65
#define Name "Seven"
#elif __GLIBC_FLT_EVAL_METHOD == 128
#define Name "Eight"
#elif __GLIBC_FLT_EVAL_METHOD == 129
#define Name "Nine"
#else
#error "Unknown __GLIBC_FLT_EVAL_METHOD"
#endif

int foo() {
  // CHECK: #define Name "One"
  // EXT: #define Name "Three"
  return Name;
}

#pragma fp eval_method(double)

#if __FLT_EVAL_METHOD__ == 3
#define Val "Unset"
#elif __FLT_EVAL_METHOD__ == 0
#define Val "val0"
#elif __FLT_EVAL_METHOD__ == 1
#define Val "val1"
#elif __FLT_EVAL_METHOD__ == 2
#define Val "val2"
#endif

int goo() {
  // CHECK: #define Val "val0"
  // EXT: #define Val "val2"
  return Val;
}
