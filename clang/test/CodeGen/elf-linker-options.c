// RUN: %clang_cc1 -triple i686---elf -emit-llvm %s -o - | FileCheck %s

#pragma comment(lib, "alpha")

// CHECK: !llvm.linker.options = !{[[NODE:![0-9]+]]}
// CHECK: [[NODE]] = !{!"lib", !"alpha"}

