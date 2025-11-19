// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s -check-prefix=DISABLED
// RUN: %clang_cc1 -fwinx64-eh-unwindv2=disabled -emit-llvm %s -o - | FileCheck %s -check-prefix=DISABLED
// RUN: %clang_cc1 -fwinx64-eh-unwindv2=best-effort -emit-llvm %s -o - | FileCheck %s -check-prefix=BESTEFFORT
// RUN: %clang_cc1 -fwinx64-eh-unwindv2=required -emit-llvm %s -o - | FileCheck %s -check-prefix=REQUIRED
// RUN: %clang -fwinx64-eh-unwindv2=best-effort -S -emit-llvm %s -o - | FileCheck %s -check-prefix=BESTEFFORT

void f(void) {}

// BESTEFFORT: !"winx64-eh-unwindv2", i32 1}
// REQUIRED: !"winx64-eh-unwindv2", i32 2}
// DISABLED-NOT: "winx64-eh-unwindv2"
