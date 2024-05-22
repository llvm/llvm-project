// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

wchar_t s[] = L"\u2722";

// CHECK: @s ={{.*}} global [2 x i32] [i32 10018, i32 0], align 4
