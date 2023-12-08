// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 -mconstructor-aliases | FileCheck %s

struct A { 
  A();
};

// CHECK: @_ZN1AC1Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN1AC2Ev
// CHECK-LABEL: define{{.*}} void @_ZN1AC2Ev(ptr {{[^,]*}} %this) unnamed_addr
A::A() { }

struct B : virtual A { 
  B();
};

// CHECK-LABEL: define{{.*}} void @_ZN1BC2Ev(ptr {{[^,]*}} %this, ptr noundef %vtt) unnamed_addr
// CHECK-LABEL: define{{.*}} void @_ZN1BC1Ev(ptr {{[^,]*}} %this) unnamed_addr
B::B() { }

struct C : virtual A {
  C(bool);
};

// CHECK-LABEL: define{{.*}} void @_ZN1CC2Eb(ptr {{[^,]*}} %this, ptr noundef %vtt, i1 noundef zeroext %0) unnamed_addr
// CHECK-LABEL: define{{.*}} void @_ZN1CC1Eb(ptr {{[^,]*}} %this, i1 noundef zeroext %0) unnamed_addr
C::C(bool) { }

// PR6251
namespace PR6251 {

// Test that we don't call the A<char> constructor twice.

template<typename T>
struct A { A(); };

struct B : virtual A<char> { };
struct C : virtual A<char> { };

struct D : B, C  {
  D();
};

// CHECK-LABEL: define{{.*}} void @_ZN6PR62511DC1Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN6PR62511AIcEC2Ev
// CHECK-NOT: call void @_ZN6PR62511AIcEC2Ev
// CHECK: ret void
D::D() { }

}

namespace virtualBaseAlignment {

// Check that the store to B::x in the base constructor has an 8-byte alignment.

// CHECK: define linkonce_odr void @_ZN20virtualBaseAlignment1BC1Ev(ptr {{[^,]*}} %[[THIS:.*]])
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]], align 8
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK: %[[X:.*]] = getelementptr inbounds %[[STRUCT_B:.*]], ptr %[[THIS1]], i32 0, i32 2
// CHECK: store i32 123, ptr %[[X]], align 16

// CHECK: define linkonce_odr void @_ZN20virtualBaseAlignment1BC2Ev(ptr {{[^,]*}} %[[THIS:.*]], ptr noundef %{{.*}})
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]], align 8
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK: %[[X:.*]] = getelementptr inbounds %[[STRUCT_B]], ptr %[[THIS1]], i32 0, i32 2
// CHECK: store i32 123, ptr %[[X]], align 8

struct A {
  __attribute__((aligned(16))) double data1;
};

struct B : public virtual A {
  B() : x(123) {}
  double a;
  int x;
};

struct C : public virtual B {};

void test() { B b; C c; }

}
