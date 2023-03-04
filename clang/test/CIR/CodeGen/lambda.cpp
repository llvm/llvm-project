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

void l0() {
  int i;
  auto a = [&](){ i = i + 1; };
  a();
}

// CHECK: cir.func lambda internal @_ZZ2l0vENK3$_0clEv(

// CHECK: %0 = cir.alloca !cir.ptr<!ty_22class2Eanon221>, cir.ptr <!cir.ptr<!ty_22class2Eanon221>>, ["this", init]
// CHECK: %1 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22class2Eanon221>>, !cir.ptr<!ty_22class2Eanon221>
// CHECK: %2 = "cir.struct_element_addr"(%1) <{member_name = "i"}> : (!cir.ptr<!ty_22class2Eanon221>) -> !cir.ptr<!cir.ptr<i32>>
// CHECK: %3 = cir.load %2 : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK: %4 = cir.load %3 : cir.ptr <i32>, i32
// CHECK: %5 = cir.cst(1 : i32) : i32
// CHECK: %6 = cir.binop(add, %4, %5) : i32
// CHECK: %7 = "cir.struct_element_addr"(%1) <{member_name = "i"}> : (!cir.ptr<!ty_22class2Eanon221>) -> !cir.ptr<!cir.ptr<i32>>
// CHECK: %8 = cir.load %7 : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK: cir.store %6, %8 : i32, cir.ptr <i32>

// CHECK: cir.func @_Z2l0v() {
