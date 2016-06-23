// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple thumbv7-windows -fms-extensions -emit-llvm -o - | FileCheck %s

#pragma detect_mismatch("test", "1")

#define BAR "2"
#pragma detect_mismatch("test2", BAR)

// CHECK: !llvm.module.flags = !{{{.*}}}
// CHECK: !{{[0-9]+}} = !{i32 6, !"Linker Options", ![[link_opts:[0-9]+]]}
// CHECK: ![[link_opts]] = !{![[test:[0-9]+]], ![[test2:[0-9]+]]}
// CHECK: ![[test]] = !{!"/FAILIFMISMATCH:\22test=1\22"}
// CHECK: ![[test2]] = !{!"/FAILIFMISMATCH:\22test2=2\22"}
