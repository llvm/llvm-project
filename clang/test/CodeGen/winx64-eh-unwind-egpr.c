// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-windows-msvc -target-feature +egpr -emit-llvm %s -o - | FileCheck %s -check-prefix=EGPR-DEFAULT
// RUN: %clang_cc1 -triple x86_64-windows-msvc -target-feature +egpr -fwinx64-eh-unwind=v3 -emit-llvm %s -o - | FileCheck %s -check-prefix=EGPR-V3
// RUN: not %clang_cc1 -triple x86_64-windows-msvc -target-feature +egpr -fwinx64-eh-unwind=v1 -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=EGPR-V1-ERROR
// RUN: not %clang_cc1 -triple x86_64-windows-msvc -target-feature +egpr -fwinx64-eh-unwind=v2-best-effort -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=EGPR-V2-ERROR
// RUN: not %clang_cc1 -triple x86_64-windows-msvc -target-feature +egpr -fwinx64-eh-unwind=v2-required -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=EGPR-V2-ERROR
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm %s -o - | FileCheck %s -check-prefix=NO-EGPR
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -target-feature +egpr -emit-llvm %s -o - | FileCheck %s -check-prefix=EGPR-LINUX

void f(void) {}

// EGPR on Windows x64 with no explicit unwind mode should auto-promote to V3.
// EGPR-DEFAULT: !"winx64-eh-unwind", i32 3}

// EGPR on Windows x64 with explicit V3 should still emit V3.
// EGPR-V3: !"winx64-eh-unwind", i32 3}

// EGPR on Windows x64 with explicit V1 should produce an error.
// EGPR-V1-ERROR: error: EGPR target feature requires unwind version 3

// EGPR on Windows x64 with explicit V2 should produce an error.
// EGPR-V2-ERROR: error: EGPR target feature requires unwind version 3

// EGPR on Windows x64 with explicit V1 should respect the user's choice.
// EGPR-EXPLICIT-V1-NOT: "winx64-eh-unwind"

// Without EGPR on Windows x64, default should be V1 (no flag emitted).
// NO-EGPR-NOT: "winx64-eh-unwind"

// EGPR on non-Windows should not emit the flag.
// EGPR-LINUX-NOT: "winx64-eh-unwind"
