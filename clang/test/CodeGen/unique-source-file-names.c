// RUN: %clang_cc1 -funique-source-file-identifier=foo -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// CHECK:  !{i32 5, !"Unique Source File Identifier", ![[MD:[0-9]*]]}
// CHECK: ![[MD]] = !{!"foo"}
