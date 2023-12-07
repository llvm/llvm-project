// RUN: %clang_cc1 -triple i686-linux-gnu -std=c++20 -emit-llvm %s -disable-llvm-passes -o - | FileCheck %s --check-prefix=CHECK-O0
// RUN: %clang_cc1 -triple i686-linux-gnu -std=c++20 -emit-llvm %s -O1 -disable-llvm-passes -o - | FileCheck %s

// Check that we add an llvm.invariant.start.p0i8 to mark when a global becomes
// read-only. If globalopt can fold the initializer, it will then mark the
// variable as constant.

// Do not produce markers at -O0.
// CHECK-O0-NOT: llvm.invariant.start.p0i8

struct A {
  A();
  int n;
};

// CHECK: @a ={{.*}} global {{.*}} zeroinitializer
extern const A a = A();

struct A2 {
  A2();
  constexpr ~A2() {}
  int n;
};

// CHECK: @a2 ={{.*}} global {{.*}} zeroinitializer
extern const A2 a2 = A2();


struct B {
  B();
  mutable int n;
};

// CHECK: @b ={{.*}} global {{.*}} zeroinitializer
extern const B b = B();

struct C {
  C();
  ~C();
  int n;
};

// CHECK: @c ={{.*}} global {{.*}} zeroinitializer
extern const C c = C();

int f();
// CHECK: @d ={{.*}} global i32 0
extern const int d = f();

void e() {
  static const A a = A();
}

// CHECK: call void @_ZN1AC1Ev(ptr noundef {{[^,]*}} @a)
// CHECK: call {{.*}}@llvm.invariant.start.p0(i64 4, ptr @a)

// CHECK: call void @_ZN2A2C1Ev(ptr noundef {{[^,]*}} @a2)
// CHECK: call {{.*}}@llvm.invariant.start.p0(i64 4, ptr @a2)

// CHECK: call void @_ZN1BC1Ev(ptr noundef {{[^,]*}} @b)
// CHECK-NOT: call {{.*}}@llvm.invariant.start.p0(i64 noundef 4, ptr @b)

// CHECK: call void @_ZN1CC1Ev(ptr noundef {{[^,]*}} @c)
// CHECK-NOT: call {{.*}}@llvm.invariant.start.p0(i64 noundef 4, ptr @c)

// CHECK: call noundef i32 @_Z1fv(
// CHECK: store {{.*}}, ptr @d
// CHECK: call {{.*}}@llvm.invariant.start.p0(i64 4, ptr @d)

// CHECK-LABEL: define{{.*}} void @_Z1ev(
// CHECK: call void @_ZN1AC1Ev(ptr noundef {{[^,]*}} @_ZZ1evE1a)
// CHECK: call {{.*}}@llvm.invariant.start.p0(i64 4, ptr {{.*}}@_ZZ1evE1a)
// CHECK-NOT: llvm.invariant.end
