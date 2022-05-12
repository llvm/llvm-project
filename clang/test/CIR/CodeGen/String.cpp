// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

class String {
  char *storage;
  long size;
  long capacity;

public:
  String() : size{0} {}
};

void test() {
  String s;
}

//      CHECK: func @_ZN6StringC2Ev
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!_22class2EString22>
// CHECK-NEXT:   cir.store %arg0, %0
// CHECK-NEXT:   %1 = cir.load %0
// CHECK-NEXT:   %2 = "cir.struct_element_addr"(%0) <{member_name = "size"}> : (!cir.ptr<!cir.ptr<!_22class2EString22>>) -> !cir.ptr<i64>
// CHECK-NEXT:   %3 = cir.cst(0 : i32) : i32
// CHECK-NEXT:   %4 = cir.cast(integral, %3 : i32), i64
// CHECK-NEXT:   cir.store %4, %2 : i64, cir.ptr <i64>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
