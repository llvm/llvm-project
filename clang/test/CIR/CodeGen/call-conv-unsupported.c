// RUN: not %clang_cc1 -triple i686-unknown-linux-gnu -fclangir -emit-cir %s -o /dev/null 2>&1 | FileCheck %s

int __attribute__((stdcall)) foo(int x) { return x; }

// CHECK: error: ClangIR code gen Not Yet Implemented: unsupported calling convention: stdcall
