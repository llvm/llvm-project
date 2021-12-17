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

//      CHECK: module  {
// CHECK-NEXT: func @baz() {
// CHECK-NEXT:   %0 = cir.alloca !cir.struct<"struct.Foo", i32, i8, !cir.struct<"struct.Bar", i32, i8>>, cir.ptr <!cir.struct<"struct.Foo", i32, i8, !cir.struct<"struct.Bar", i32, i8>>>, [uninitialized] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.alloca !cir.struct<"struct.Bar", i32, i8>, cir.ptr <!cir.struct<"struct.Bar", i32, i8>>, [uninitialized] {alignment = 4 : i64}
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
