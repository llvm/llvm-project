// RUN: %clang_cc1 -O3 %s -emit-llvm -o - | FileCheck %s

typedef int int2 __attribute((ext_vector_type(2)));

int test1()
{
  int2 a = (int2)(1,0);
  int2 b = (int2)(1,1);
  return (a&&b).x + (a||b).y;
  // CHECK: ret i32 -2
}

int test2()
{
  int2 a = (int2)(1,0);
  return (!a).y;
  // CHECK: ret i32 -1
}

