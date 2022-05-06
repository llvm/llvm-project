// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

void fn() {
  auto a = [](){};
}

//      CHECK: !22class2Eanon22 = type !cir.struct<"class.anon", i8>
// CHECK-NEXT: module
// CHECK-NEXT:   func @_Z2fnv()
// CHECK-NEXT:     %0 = cir.alloca !22class2Eanon22, cir.ptr <!22class2Eanon22>, ["a", uninitialized]
