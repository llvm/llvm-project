// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-windows-msvc -target-feature +egpr -emit-llvm %s -o - | FileCheck %s -check-prefix=EGPR-DEFAULT
// RUN: %clang_cc1 -triple x86_64-windows-msvc -target-feature +egpr -fwinx64-eh-unwind=v3 -emit-llvm %s -o - | FileCheck %s -check-prefix=EGPR-V3
// RUN: not %clang_cc1 -triple x86_64-windows-msvc -target-feature +egpr -fwinx64-eh-unwind=v1 -fexceptions -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=EGPR-V1-ERROR
// RUN: not %clang_cc1 -triple x86_64-windows-msvc -target-feature +egpr -fwinx64-eh-unwind=v2-best-effort -fexceptions -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=EGPR-V2-ERROR
// RUN: not %clang_cc1 -triple x86_64-windows-msvc -target-feature +egpr -fwinx64-eh-unwind=v2-required -fexceptions -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=EGPR-V2-ERROR
// Per-function check should NOT fire when the function is nounwind (no unwind
// info is emitted) — this run compiles successfully without -fexceptions.
// RUN: %clang_cc1 -triple x86_64-windows-msvc -target-feature +egpr -fwinx64-eh-unwind=v1 -emit-llvm %s -o - | FileCheck %s -check-prefix=EGPR-V1-NOUNWIND
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm %s -o - | FileCheck %s -check-prefix=NO-EGPR
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -target-feature +egpr -emit-llvm %s -o - | FileCheck %s -check-prefix=EGPR-LINUX

void g(void);
void f(void) { g(); }

// EGPR on Windows x64 with no explicit unwind mode should auto-promote to V3.
// EGPR-DEFAULT: !"winx64-eh-unwind", i32 3}

// EGPR on Windows x64 with explicit V3 should still emit V3.
// EGPR-V3: !"winx64-eh-unwind", i32 3}

// EGPR on Windows x64 with explicit V1 + exceptions should produce an error
// (the function needs unwind info).
// EGPR-V1-ERROR: error: EGPR target feature requires unwind version 3

// EGPR on Windows x64 with explicit V2 + exceptions should produce an error.
// EGPR-V2-ERROR: error: EGPR target feature requires unwind version 3

// EGPR on Windows x64 with explicit V1 and no exceptions: the function is
// nounwind, so no error and no module flag (V1 is the default).
// EGPR-V1-NOUNWIND-NOT: error
// EGPR-V1-NOUNWIND-NOT: "winx64-eh-unwind"

// Without EGPR on Windows x64, default should be V1 (no flag emitted).
// NO-EGPR-NOT: "winx64-eh-unwind"

// EGPR on non-Windows should not emit the flag.
// EGPR-LINUX-NOT: "winx64-eh-unwind"
