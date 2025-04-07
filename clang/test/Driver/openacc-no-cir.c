// RUN: not %clang -fopenacc %s 2>&1 | FileCheck %s -check-prefix=ERROR
// RUN: not %clang -fclangir -fopenacc %s 2>&1 | FileCheck %s -check-prefix=NOERROR
// RUN: not %clang -fopenacc -fclangir %s 2>&1 | FileCheck %s -check-prefix=NOERROR

// ERROR: OpenACC directives will result in no runtime behavior, use -fclangir to enable runtime effect
// NOERROR-NOT: OpenACC directives will result in no runtime behavior, use -fclangir to enable runtime effect
