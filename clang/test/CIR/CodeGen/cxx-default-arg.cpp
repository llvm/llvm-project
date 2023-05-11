// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: cir.func private @_ZN12MyIntPointerC1EPi

struct MyIntPointer {
  MyIntPointer(int *p = nullptr);
};

void foo() {
  MyIntPointer p;
}