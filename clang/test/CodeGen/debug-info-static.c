// RUN: %clang_cc1  -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

// CHECK: !DIGlobalVariable({{.*}}variable: i32* @f.xyzzy
void f(void)
{
   static int xyzzy;
   xyzzy += 3;
}
