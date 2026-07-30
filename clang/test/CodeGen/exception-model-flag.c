// Verify clang records the "exception-model" module flag when the exception
// model differs from the target triple's default, and omits it otherwise.
// The cc1 -exception-model option accepts dwarf/sjlj/seh/wasm/none; ARM EHABI
// is triple-inferred and not user-selectable here.

// i686-linux defaults to DWARF exception handling, so every other model emits.
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=sjlj -emit-llvm %s -o - | FileCheck %s --check-prefix=SJLJ
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=seh -emit-llvm %s -o - | FileCheck %s --check-prefix=WINEH

// SEH maps to the "wineh" spelling regardless of the requesting triple.
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=seh -emit-llvm %s -o - | FileCheck %s --check-prefix=WINEH

// Wasm EH (needs the backend enable flag) records the "wasm" model.
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -fexceptions -exception-model=wasm -mllvm -wasm-enable-eh -emit-llvm %s -o - | FileCheck %s --check-prefix=WASM

// DWARF requested on a target that also defaults to DWARF: no flag.
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=dwarf -emit-llvm %s -o - | FileCheck %s --check-prefix=NONE

// SjLj requested on a target that defaults to SjLj: no flag.
// RUN: %clang_cc1 -triple armv7-apple-ios -fexceptions -exception-model=sjlj -emit-llvm %s -o - | FileCheck %s --check-prefix=NONE

// A target that defaults to WinEH records no flag (clang rejects an explicit
// -exception-model=seh here, so rely on the default).
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexceptions -emit-llvm %s -o - | FileCheck %s --check-prefix=NONE

// Explicitly "none": no flag (clang cannot distinguish this from unspecified).
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=none -emit-llvm %s -o - | FileCheck %s --check-prefix=NONE

// No exception model requested at all: no flag.
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=NONE
// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabi -emit-llvm %s -o - | FileCheck %s --check-prefix=NONE

void f(void) {}

// SJLJ: !{i32 1, !"exception-model", !"sjlj"}
// WINEH: !{i32 1, !"exception-model", !"wineh"}
// WASM: !{i32 1, !"exception-model", !"wasm"}
// NONE-NOT: "exception-model"
