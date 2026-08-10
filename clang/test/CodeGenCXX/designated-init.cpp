// RUN: %clang_cc1 -std=c++98 -emit-llvm -o - %s -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm -o - %s -triple x86_64-linux-gnu | FileCheck %s

struct A { int x, y[3]; };
struct B { A a; };

// CHECK: @b ={{.*}} global %{{[^ ]*}} { %{{[^ ]*}} { i32 1, [3 x i32] [i32 2, i32 5, i32 4] } }
B b = {(A){1, 2, 3, 4}, .a.y[1] = 5};

union U {
  int n;
  float f;
};
struct C {
  int x;
  U u[3];
};
struct D {
  C c;
};

// CHECK: @d1 = {{.*}} { i32 1, [3 x %[[U:.*]]] [%[[U]] { i32 2 }, %[[U]] { i32 5 }, %[[U]] { i32 4 }] }
D d1 = {(C){1, {{.n=2}, {.f=3}, {.n=4}}}, .c.u[1].n = 5};

// CHECK: @d2 = {{.*}} { i32 1, { %[[U]], float, %[[U]] } { %[[U]] { i32 2 }, float 5.{{0*}}e+00, %[[U]] { i32 4 } } }
D d2 = {(C){1, 2, 3, 4}, .c.u[1].f = 5};

struct Bitfield {
  int a : 3;
  int b : 4;
  int c : 5;
};
struct WithBitfield {
  int n;
  Bitfield b;
};
// CHECK: @bitfield = {{.*}} { i32 1, { i8, i8, [2 x i8] } { i8 42, i8 2, [2 x i8] undef } }
WithBitfield bitfield = {1, (Bitfield){2, 3, 4}, .b.b = 5};

struct String {
  const char buffer[12];
};
struct WithString {
  String str;
};
// CHECK: @string = {{.*}} [12 x i8] c"Hello World\00" } }
WithString string = {(String){"hello world"}, .str.buffer[0] = 'H', .str.buffer[6] = 'W'};

struct LargeArray {
  int arr[4096];
};
struct WithLargeArray {
  LargeArray arr;
};
// CHECK: @large ={{.*}} global { { <{ [11 x i32], [4085 x i32] }> } } { { <{ [11 x i32], [4085 x i32] }> } { <{ [11 x i32], [4085 x i32] }> <{ [11 x i32] [i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 10], [4085 x i32] zeroinitializer }> } }
WithLargeArray large = {(LargeArray){1, 2, 3}, .arr.arr[10] = 10};

union OverwritePaddingWithBitfield {
  struct Padding { unsigned : 8; char c; } padding;
  char bitfield : 3;
};
struct WithOverwritePaddingWithBitfield {
  OverwritePaddingWithBitfield a;
};
// CHECK: @overwrite_padding ={{.*}} global { { i8, i8 } } { { i8, i8 } { i8 3, i8 1 } }
WithOverwritePaddingWithBitfield overwrite_padding = {(OverwritePaddingWithBitfield){1}, .a.bitfield = 3};
