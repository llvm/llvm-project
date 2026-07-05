// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s

void foo(void) {
  static int hex = 0x10;
  static int dec = 10;
}
void bar(void) {
  static int hex = 0x20;
  static int dec = 20;
}
