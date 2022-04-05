// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm %s -o - | FileCheck %s

// CHECK: @constinit = private global [3 x i8*] [i8* blockaddress(@main, %L), i8* null, i8* null]

void receivePtrs(void **);

int main(void) {
L:
  receivePtrs((void *[]){ &&L, 0, 0 });
}
