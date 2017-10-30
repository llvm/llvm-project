// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s
//
// Check that we generate correct TBAA information for reference accesses.

struct S;

struct B {
  S &s;
  B(S &s);
  S &get();
};

B::B(S &s) : s(s) {
// CHECK-LABEL: _ZN1BC2ER1S
// Check initialization of the reference parameter.
// CHECK: store %struct.S* {{.*}}, %struct.S** {{.*}}, !tbaa [[TAG_pointer:!.*]]

// Check loading of the reference parameter.
// CHECK: load %struct.S*, %struct.S** {{.*}}, !tbaa [[TAG_pointer]]

// Check initialization of the reference member.
// CHECK: store %struct.S* {{.*}}, %struct.S** {{.*}}, !tbaa [[TAG_pointer]]
}

S &B::get() {
// CHECK-LABEL: _ZN1B3getEv
// Check that we access the reference as a structure member.
// CHECK: load %struct.S*, %struct.S** {{.*}}, !tbaa [[TAG_B_s:!.*]]
  return s;
}

// CHECK-DAG: [[TAG_pointer]] = !{[[TYPE_pointer:!.*]], [[TYPE_pointer]], i64 0}
// CHECK-DAG: [[TAG_B_s]] = !{[[TYPE_B:!.*]], [[TYPE_pointer]], i64 0}
//
// CHECK-DAG: [[TYPE_B]] = !{!"_ZTS1B", [[TYPE_pointer]], i64 0}
// CHECK-DAG: [[TYPE_pointer]] = !{!"any pointer", [[TYPE_char:!.*]], i64 0}
// CHECK-DAG: [[TYPE_char]] = !{!"omnipotent char", {{!.*}}, i64 0}
