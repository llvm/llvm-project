// RUN: %clang_cc1 -funique-source-file-names -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// CHECK:  !{i32 7, !"Unique Source File Names", i32 1}
