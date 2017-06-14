// RUN: %clang_cc1 %s --dependent-lib=msvcrt -triple thumbv7-windows -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s --dependent-lib=msvcrt -triple i686-pc-win32 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s --dependent-lib=msvcrt -triple x86_64-pc-win32 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s --dependent-lib=msvcrt -triple i686-pc-linux -emit-llvm -o - | FileCheck -check-prefix LINUX %s

// CHECK: !llvm.linker.options = !{![[msvcrt:[0-9]+]]}
// CHECK: ![[msvcrt]] = !{!"/DEFAULTLIB:msvcrt.lib"}

// LINUX: !llvm.linker.options = !{![[msvcrt:[0-9]+]]}
// LINUX: ![[msvcrt]] = !{!"-lmsvcrt"}

int f();
