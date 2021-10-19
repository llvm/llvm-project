// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s

void foo() {
  static int i, j;
}
void bar() {
  static int i, j;
}
