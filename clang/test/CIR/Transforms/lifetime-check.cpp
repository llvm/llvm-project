// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-tool %t.cir -cir-lifetime-check -o %t-out.cir 2>&1 | FileCheck %s

int *basic() {
  int *p = nullptr;
  {
    int x = 0;
    p = &x;
    *p = 42;
  }
  *p = 42;
  return p;
}

// CHECK: Hello Lifetime World