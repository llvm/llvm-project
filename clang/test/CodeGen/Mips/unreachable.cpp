// REQUIRES: mips-registered-target
// RUN: %clang_cc1 -triple mipsel-w64-windows-gnu -x c++ -mrelocation-model static -emit-obj %s -o - | llvm-objdump -a - | FileCheck %s
// CHECK: file format coff-mips

[[__noreturn__]] inline void g() {
  __builtin_unreachable();
}

void f(int i)
{
  if (i == 0)
    g();
}
