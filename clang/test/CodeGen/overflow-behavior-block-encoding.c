// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -fblocks -emit-llvm -o - %s | FileCheck %s

// A block signature encodes an overflow behavior type as its underlying type.

// CHECK: c"i8@?0\00"
int ret(int n) {
  int __ob_trap a = n;
  int (^b)(void) = ^{ return a + 1; };
  return b();
}

// CHECK: c"v12@?0I8\00"
void param(void) {
  void (^b)(unsigned __ob_wrap) = ^(unsigned __ob_wrap x) {};
  b(1);
}
