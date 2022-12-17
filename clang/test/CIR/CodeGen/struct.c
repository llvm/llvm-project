// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Bar {
  int a;
  char b;
};

struct Foo {
  int a;
  char b;
  struct Bar z;
};

void baz() {
  struct Bar b;
  struct Foo f;
}

//      CHECK: !ty_22struct2EBar22 = !cir.struct<"struct.Bar", i32, i8>
// CHECK-NEXT: !ty_22struct2EFoo22 = !cir.struct<"struct.Foo", i32, i8, !ty_22struct2EBar22>
//  CHECK-DAG: module {{.*}} {
// CHECK-NEXT:   cir.func @baz() {
// CHECK-NEXT:     %0 = cir.alloca !ty_22struct2EBar22, cir.ptr <!ty_22struct2EBar22>, ["b"] {alignment = 4 : i64}
// CHECK-NEXT:     %1 = cir.alloca !ty_22struct2EFoo22, cir.ptr <!ty_22struct2EFoo22>, ["f"] {alignment = 4 : i64}
// CHECK-NEXT:     cir.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
