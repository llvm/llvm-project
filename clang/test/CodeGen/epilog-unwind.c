// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s -check-prefix=DISABLED
// RUN: %clang_cc1 -fwinx64-eh-unwindv2 -emit-llvm %s -o - | FileCheck %s -check-prefix=ENABLED
// RUN: %clang -fwinx64-eh-unwindv2 -S -emit-llvm %s -o - | FileCheck %s -check-prefix=ENABLED
// RUN: %clang -fno-winx64-eh-unwindv2 -S -emit-llvm %s -o - | FileCheck %s -check-prefix=DISABLED

void f(void) {}

// ENABLED: !"winx64-eh-unwindv2", i32 1}
// DISABLED-NOT: "winx64-eh-unwindv2"
