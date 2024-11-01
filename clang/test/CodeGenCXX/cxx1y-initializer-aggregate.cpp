// RUN: %clang_cc1 -std=c++1y %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s

struct A {
  int n = 0;
  const char *p;
  char k = p[n];
  int f();
  int x = f();
  union {
    char c;
    double d = 1.0;
  };
};

int f();

union B {
  int a;
  int f();
  int b = f();
};

A a { .p = "foobar" };
A b { 4, "bazquux", .x = 42, .c = 9 };
A c { 1, 0, 'A', f(), { 3 } };

// CHECK: @[[STR_A:.*]] = {{.*}} [7 x i8] c"foobar\00"
// CHECK: @a ={{.*}} global {{.*}} zeroinitializer

// @b has a constant initializer
// CHECK: @[[STR_B:.*]] = {{.*}} [8 x i8] c"bazquux\00"
// CHECK: @b ={{.*}} global {{.*}} i32 4, {{.*}} @[[STR_B]], i8 117, i32 42, {{.*}} i8 9

B x;
B y {};
B z { 1 };
// CHECK: @z ={{.*}} global {{.*}} { i32 1 }

// Brace initialization should initialize the first field even though it is
// unnamed.
union C {
  struct {
    int C::*memptr;
  };
};

C n{};
// CHECK: @n ={{.*}} global %union.C { %struct.anon { i64 -1 } }, align 8

// Initialization of 'a':

// CHECK: store i32 0, ptr @a
// CHECK: store ptr @[[STR_A]], ptr getelementptr inbounds ({{.*}} @a, i32 0, i32 1)
// CHECK: load ptr, ptr getelementptr inbounds ({{.*}} @a, i32 0, i32 1)
// CHECK: load i32, ptr @a
// CHECK: getelementptr inbounds i8, ptr %{{.*}}, {{.*}} %{{.*}}
// CHECK: store i8 %{{.*}}, ptr getelementptr inbounds ({{.*}} @a, i32 0, i32 2)
// CHECK: call noundef i32 @_ZN1A1fEv({{.*}} @a)
// CHECK: store i32 %{{.*}}, ptr getelementptr inbounds ({{.*}}, ptr @a, i32 0, i32 3)
// CHECK: store double 1.000000e+00, ptr getelementptr inbounds ({{.*}} @a, i32 0, i32 4)

// No dynamic initialization of 'b':

// CHECK-NOT: @b

// Initialization of 'c':

// CHECK: store i32 1, ptr @c
// CHECK: store ptr null, ptr getelementptr inbounds ({{.*}} @c, i32 0, i32 1)
// CHECK-NOT: load
// CHECK: store i8 65, ptr getelementptr inbounds ({{.*}} @c, i32 0, i32 2)
// CHECK: call noundef i32 @_Z1fv()
// CHECK: store i32 %{{.*}}, ptr getelementptr inbounds ({{.*}}, ptr @c, i32 0, i32 3)
// CHECK-NOT: C1Ev
// CHECK: store i8 3, ptr {{.*}} @c, i32 0, i32 4)

// CHECK: call void @_ZN1BC1Ev({{.*}} @x)

// CHECK: call noundef i32 @_ZN1B1fEv({{.*}} @y)
// CHECK: store i32 %{{.*}}, ptr @y
