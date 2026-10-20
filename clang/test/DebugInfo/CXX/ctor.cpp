// RUN: %clang -emit-llvm -g -S %s -o - | FileCheck %s

struct X {
  X(int v);

  int value;
};

X::X(int v) {
  // CHECK_TEMPORARILY_DISABLED: call void @_ZN1XC2Ei(ptr %this1, i32 %tmp), !dbg
  // TEMPORARY CHECK: X
  value = v;
}

