// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Foo {
  int a;
  char b;
};

void bar() {
  struct Foo f;
}

//      CHECK: module  {
// CHECK-NEXT: func @bar() {
// CHECK-NEXT:   %0 = cir.alloca !cir.struct<"struct.Foo", i32, i8>, cir.ptr <!cir.struct<"struct.Foo", i32, i8>>, [uninitialized] {alignment = 4 : i64}
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
