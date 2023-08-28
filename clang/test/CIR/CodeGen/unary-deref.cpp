// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

struct MyIntPointer {
  int *ptr = nullptr;
  int read() const { return *ptr; }
};

void foo() {
  MyIntPointer p;
  (void)p.read();
}

// CHECK:  cir.func linkonce_odr  @_ZNK12MyIntPointer4readEv
// CHECK:  %2 = cir.load %0
// CHECK:  %3 = cir.get_member %2[0] {name = "ptr"}
// CHECK:  %4 = cir.load deref %3 : cir.ptr <!cir.ptr<!s32i>>
// CHECK:  %5 = cir.load %4