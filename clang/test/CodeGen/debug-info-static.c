// RUN: %clang_cc1  -g -emit-llvm -o - %s | FileCheck %s

// CHECK:  xyzzy} ; [ DW_TAG_variable ]
void f(void)
{
   static int xyzzy;
   xyzzy += 3;
}
