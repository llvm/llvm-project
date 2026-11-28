// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7.0 -emit-llvm -o - %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-apple-macosx10.7.0 -emit-llvm -o - %s | FileCheck %s %}
// CHECK: @b ={{.*}} thread_local global i32 5, align 4
__thread int b = 5;
