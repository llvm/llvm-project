// RUN: %clang -fopenacc -S %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=ERROR
// RUN: %clang -fclangir -fopenacc -S %s -o /dev/null 2>&1 | FileCheck %s --allow-empty -check-prefix=NOERROR
// RUN: %clang -fopenacc -fclangir -S %s -o /dev/null 2>&1 | FileCheck %s --allow-empty -check-prefix=NOERROR

// ERROR: OpenACC directives will result in no runtime behavior; use -fclangir to enable runtime effect
// NOERROR-NOT: OpenACC directives
