// RUN: %clang_cc1 %s --linker-option=/include:foo -triple i686-pc-win32 -emit-llvm -o - | FileCheck %s

// CHECK: !llvm.module.flags = !{{{.*}}}
// CHECK: !{{[0-9]+}} = !{i32 6, !"Linker Options", ![[link_opts:[0-9]+]]}
// CHECK: ![[link_opts]] = !{![[msvcrt:[0-9]+]]}
// CHECK: ![[msvcrt]] = !{!"/include:foo"}

int f();
