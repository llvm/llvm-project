// RUN: %clang_cc1 -triple x86_64-linux -std=c++26 -fexperimental-new-constant-interpreter %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c++26                                         %s -emit-llvm -o - | FileCheck %s

struct A { int a; };
struct B : virtual A {
  constexpr B() : A(127) {}
};


// CHECK: @b1 =  global { ptr, i32 } { {{.*}} i32 127 }
B b1{};

struct C : B {
  constexpr C(int) : B(), A(128) {}
};

// CHECK: @c1 =  global { ptr, i32 } { {{.*}} i32 128 }
C c1 = C(12);

// CHECK: @a1 = global ptr getelementptr (i8, {{.*}}, i64 8)
A* a1 = (A*)&c1;

struct X {
  char x[4] = {};
};
struct Y {
  char y[4] = {};
};
struct Z : virtual X, virtual Y {
  char z[4] = {};
};

struct W : Z {};


// CHECK: @w = global {{.*}}
W w;
// CHECK: @z = constant ptr @w
Z &z = w;
// CHECK: @x = constant ptr getelementptr (i8, ptr @w, i64 12)
X &x = z;
// CHECK: @y = constant ptr getelementptr (i8, ptr @w, i64 16)
Y &y = z;
