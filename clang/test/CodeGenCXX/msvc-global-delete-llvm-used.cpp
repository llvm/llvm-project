// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=x86_64-pc-windows-msvc -o - \
// RUN:   | FileCheck %s --implicit-check-not="@llvm.used.1"

// The __empty_global_delete fallback is marked used so it is always emitted.
// It must join the single llvm.used that CodeGenModule emits at end-of-TU: if
// it creates its own llvm.used first, the one holding __attribute__((used))
// globals gets renamed to llvm.used.1, which LLVM ignores.

struct S { virtual ~S(); };
S::~S() {}
void del(S *s) { ::delete s; }

__attribute__((used)) static void keep_me() {}

// CHECK: @llvm.used = appending global
// CHECK-SAME: @"?__empty_global_delete@@YAXPEAX_K@Z"
// CHECK-SAME: @"?keep_me@@YAXXZ"
