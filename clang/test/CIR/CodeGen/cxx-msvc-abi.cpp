// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

void empty() {}

// CIR: cir.func {{.*}} @"?empty@@YAXXZ"()
