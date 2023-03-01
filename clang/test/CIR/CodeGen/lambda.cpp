// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void fn() {
  auto a = [](){};
  a();
}

//      CHECK: !ty_22class2Eanon22 = !cir.struct<"class.anon", i8>
//  CHECK-DAG: module

//      CHECK: cir.func lambda internal @_ZZ2fnvENK3$_0clEv

//      CHECK:   cir.func @_Z2fnv()
// CHECK-NEXT:     %0 = cir.alloca !ty_22class2Eanon22, cir.ptr <!ty_22class2Eanon22>, ["a"]
//      CHECK:   cir.call @_ZZ2fnvENK3$_0clEv
