// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | opt -S | FileCheck %s

void f(void) {}
