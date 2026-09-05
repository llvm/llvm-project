// Verify clang records the "exception-model" module flag whenever an exception
// model is specified on the command line, and omits it only when the model is
// left unspecified. The flag is emitted even when the requested model matches
// the target triple's default, so that its absence unambiguously means
// "unspecified" and conflicting models are rejected at link time.
// The cc1 -exception-model option accepts dwarf/sjlj/seh/wasm/none; ARM EHABI
// is triple-inferred and not user-selectable here.

// i686-linux defaults to DWARF exception handling.
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=sjlj -emit-llvm %s -o - | FileCheck %s --check-prefix=SJLJ
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=seh -emit-llvm %s -o - | FileCheck %s --check-prefix=WINEH

// SEH maps to the "wineh" spelling regardless of the requesting triple.
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=seh -emit-llvm %s -o - | FileCheck %s --check-prefix=WINEH

// Wasm EH (needs the backend enable flag) records the "wasm" model.
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -fexceptions -exception-model=wasm -mllvm -wasm-enable-eh -emit-llvm %s -o - | FileCheck %s --check-prefix=WASM

// A requested model that matches the target default is still recorded, so that
// the flag's absence always means "unspecified".
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=dwarf -emit-llvm %s -o - | FileCheck %s --check-prefix=DWARF
// RUN: %clang_cc1 -triple armv7-apple-ios -fexceptions -exception-model=sjlj -emit-llvm %s -o - | FileCheck %s --check-prefix=SJLJ

// Explicitly "none" disables exceptions and is recorded as such.
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=none -emit-llvm %s -o - | FileCheck %s --check-prefix=NONE

// No exception model requested at all: unspecified, so no flag.
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=UNSPEC
// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabi -emit-llvm %s -o - | FileCheck %s --check-prefix=UNSPEC
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexceptions -emit-llvm %s -o - | FileCheck %s --check-prefix=UNSPEC

void f(void) {}

// SJLJ: !{i32 1, !"exception-model", !"sjlj"}
// WINEH: !{i32 1, !"exception-model", !"wineh"}
// WASM: !{i32 1, !"exception-model", !"wasm"}
// DWARF: !{i32 1, !"exception-model", !"dwarf"}
// NONE: !{i32 1, !"exception-model", !"none"}
// UNSPEC-NOT: "exception-model"
