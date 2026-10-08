// RUN: %clang -fopenacc -S %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=ERROR
// RUN: %clang -fclangir -fopenacc -S -target=x86_64-linux-gnu %s -o /dev/null 2>&1 | FileCheck %s --allow-empty -check-prefix=NOERROR
// RUN: %clang -fopenacc -fclangir -S -target=x86_64-linux-gnu %s -o /dev/null 2>&1 | FileCheck %s --allow-empty -check-prefix=NOERROR

// ERROR: OpenACC directives will result in no runtime behavior; use -fclangir to enable runtime effect
// NOERROR-NOT: OpenACC directives
