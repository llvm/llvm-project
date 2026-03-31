// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: cir.func {{.*}}@_Z5emptyv()
void empty() {}
