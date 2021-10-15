// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -std=c++11 -emit-llvm -o - %s | opt -instnamer -S | FileCheck %s

struct A {
  A();
  A(const A&);
  ~A();
};

struct B {
  B();
  B(const B&);
  ~B();
};

struct C {
  C();
  C(const C&);
  ~C();
};

struct D {
  D();
  D(const D&);
  ~D();

  B *begin();
  B *end();
};

B *begin(C&);
B *end(C&);

extern B array[5];

// CHECK-LABEL: define{{.*}} void @_Z9for_arrayv(
void for_array() {
  // CHECK: call void @_ZN1AC1Ev(%struct.A* noundef [[A:.*]])
  A a;
  for (B b : array) {
    // CHECK-NOT: 5begin
    // CHECK-NOT: 3end
    // CHECK: getelementptr {{.*}}, i64 0
    // CHECK: getelementptr {{.*}}, i64 5
    // CHECK: br label %[[COND:.*]]

    // CHECK: [[COND]]:
    // CHECK: %[[CMP:.*]] = icmp ne
    // CHECK: br i1 %[[CMP]], label %[[BODY:.*]], label %[[END:.*]]

    // CHECK: [[BODY]]:
    // CHECK: call void @_ZN1BC1ERKS_(
    // CHECK: call void @_ZN1BD1Ev(
    // CHECK: br label %[[INC:.*]]

    // CHECK: [[INC]]:
    // CHECK: getelementptr {{.*}} i32 1
    // CHECK: br label %[[COND]]
  }
  // CHECK: [[END]]:
  // CHECK: call void @_ZN1AD1Ev(%struct.A* noundef [[A]])
  // CHECK: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z9for_rangev(
void for_range() {
  // CHECK: call void @_ZN1AC1Ev(%struct.A* noundef [[A:.*]])
  A a;
  for (B b : C()) {
    // CHECK: call void @_ZN1CC1Ev(
    // CHECK: = call noundef %struct.B* @_Z5beginR1C(
    // CHECK: = call noundef %struct.B* @_Z3endR1C(
    // CHECK: br label %[[COND:.*]]

    // CHECK: [[COND]]:
    // CHECK: %[[CMP:.*]] = icmp ne
    // CHECK: br i1 %[[CMP]], label %[[BODY:.*]], label %[[CLEANUP:.*]]

    // CHECK: [[CLEANUP]]:
    // CHECK: call void @_ZN1CD1Ev(
    // CHECK: br label %[[END:.*]]

    // CHECK: [[BODY]]:
    // CHECK: call void @_ZN1BC1ERKS_(
    // CHECK: call void @_ZN1BD1Ev(
    // CHECK: br label %[[INC:.*]]

    // CHECK: [[INC]]:
    // CHECK: getelementptr {{.*}} i32 1
    // CHECK: br label %[[COND]]
  }
  // CHECK: [[END]]:
  // CHECK: call void @_ZN1AD1Ev(%struct.A* noundef [[A]])
  // CHECK: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z16for_member_rangev(
void for_member_range() {
  // CHECK: call void @_ZN1AC1Ev(%struct.A* noundef [[A:.*]])
  A a;
  for (B b : D()) {
    // CHECK: call void @_ZN1DC1Ev(
    // CHECK: = call noundef %struct.B* @_ZN1D5beginEv(
    // CHECK: = call noundef %struct.B* @_ZN1D3endEv(
    // CHECK: br label %[[COND:.*]]

    // CHECK: [[COND]]:
    // CHECK: %[[CMP:.*]] = icmp ne
    // CHECK: br i1 %[[CMP]], label %[[BODY:.*]], label %[[CLEANUP:.*]]

    // CHECK: [[CLEANUP]]:
    // CHECK: call void @_ZN1DD1Ev(
    // CHECK: br label %[[END:.*]]

    // CHECK: [[BODY]]:
    // CHECK: call void @_ZN1BC1ERKS_(
    // CHECK: call void @_ZN1BD1Ev(
    // CHECK: br label %[[INC:.*]]

    // CHECK: [[INC]]:
    // CHECK: getelementptr {{.*}} i32 1
    // CHECK: br label %[[COND]]
  }
  // CHECK: [[END]]:
  // CHECK: call void @_ZN1AD1Ev(%struct.A* noundef [[A]])
  // CHECK: ret void
}
