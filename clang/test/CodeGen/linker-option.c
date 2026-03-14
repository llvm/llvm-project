// RUN: %clang_cc1 %s --linker-option=/include:foo -triple i686-pc-win32 -emit-llvm -o - | FileCheck %s

// CHECK: !llvm.linker.options = !{![[msvcrt:[0-9]+]]}
// CHECK: ![[msvcrt]] = !{!"/include:foo"}

int f(void);
