// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s

struct X { };

// CHECK: @x1 = {{(dso_local )?}}global %struct.X zeroinitializer
// CHECK: @x4 = {{(dso_local )?}}global %struct.X zeroinitializer
// CHECK: @x2 = external {{(dso_local )?}}global %struct.X
// CHECK: @x3 = external {{(dso_local )?}}global %struct.X
extern "C" {
  X x1;
}

extern "C" X x2;

extern X x3;

X x4;

X& get(int i) {
  if (i == 1)
    return x1;
  else if (i == 2)
    return x2;
  else if (i == 3)
    return x3;
  else
    return x4;
}
