// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int a = 3;

// CHECK: module  {
// CHECK-NEXT:   cir.global @a : i32 = 3