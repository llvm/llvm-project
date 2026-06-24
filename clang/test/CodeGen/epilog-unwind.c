// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s -check-prefix=DISABLED
// RUN: %clang_cc1 -fwinx64-eh-unwind=v1 -emit-llvm %s -o - | FileCheck %s -check-prefix=DISABLED
// RUN: %clang_cc1 -fwinx64-eh-unwind=v2-best-effort -emit-llvm %s -o - | FileCheck %s -check-prefix=BESTEFFORT
// RUN: %clang_cc1 -fwinx64-eh-unwind=v2-required -emit-llvm %s -o - | FileCheck %s -check-prefix=REQUIRED
// RUN: %clang_cc1 -fwinx64-eh-unwind=v3 -emit-llvm %s -o - | FileCheck %s -check-prefix=V3
// RUN: %clang -fwinx64-eh-unwind=v2-best-effort -S -emit-llvm %s -o - | FileCheck %s -check-prefix=BESTEFFORT

void f(void) {}

// BESTEFFORT: !"winx64-eh-unwind", i32 1}
// REQUIRED: !"winx64-eh-unwind", i32 2}
// V3: !"winx64-eh-unwind", i32 3}
// DISABLED-NOT: "winx64-eh-unwind"
