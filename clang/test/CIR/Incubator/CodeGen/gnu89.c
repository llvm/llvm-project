// RUN: %clang_cc1 -std=gnu89 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void foo() {}
//CHECK: cir.func {{.*@foo}}