// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s

// A by-value argument built by a trivial copy/move constructor from a
// variable forwards the variable's storage to the call; the byval boundary
// makes the copy, so no agg.tmp is materialized. This is the C++ analog of
// the LValueToRValue forwarding the C path already does.

struct Triv {
  unsigned long a, b, c, d;
};
void sink(Triv);

// Same-type source: forward, no temporary.
void from_local() {
  Triv t;
  sink(t);
}
// CHECK-LABEL: define {{.*}}@_Z10from_localv(
// CHECK:     [[T:%.*]] = alloca %struct.Triv
// CHECK-NOT: = alloca %struct.Triv
// CHECK:     call void @_Z4sink4Triv(ptr noundef byval(%struct.Triv) align 8 [[T]])

void from_param(Triv t) {
  sink(t);
}
// CHECK-LABEL: define {{.*}}@_Z10from_param4Triv(
// CHECK-NOT: = alloca %struct.Triv
// CHECK:     call void @_Z4sink4Triv(ptr noundef byval(%struct.Triv) align 8 %t)

// Derived-to-base: forwarding the derived lvalue into a base-typed slot would
// slice at the wrong offset, so the same-type guard keeps the temp and copies
// from the adjusted base offset.
struct Base1 {
  long a, b;
};
struct Base2 {
  long c, d;
};
struct Derived : Base1, Base2 {
  long e;
};
void sink_base(Base2);
void slice_to_base(Derived d) {
  sink_base(d);
}
// CHECK-LABEL: define {{.*}}@_Z13slice_to_base7Derived(
// CHECK:     [[TMP:%.*]] = alloca %struct.Base2
// CHECK:     [[ADJ:%.*]] = getelementptr inbounds i8, ptr %d, i64 16
// CHECK:     call void @llvm.memcpy{{.*}}(ptr {{.*}} [[TMP]], ptr {{.*}} [[ADJ]],
// CHECK:     call void @_Z9sink_base5Base2(
