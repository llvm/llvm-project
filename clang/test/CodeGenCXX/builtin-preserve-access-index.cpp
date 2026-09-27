// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=limited -disable-llvm-passes %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64 -emit-llvm -debug-info-kind=limited -disable-llvm-passes %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=constructor -disable-llvm-passes %s -o - | FileCheck %s

struct Base {
  Base();
  int base;
};
struct Record : Base {
  static int first;
  int field;
  static int second;
  unsigned bits : 3;
  void method();
};
union Union {
  int first;
  static int member;
  long second;
};
struct Dynamic {
  virtual void method();
  int field;
};

int *field(Record *p) {
  return __builtin_preserve_access_index(&p->field);
}
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0({{.*}}, i32 1, i32 2), {{.*}}!llvm.preserve.access.index ![[RECORD:[0-9]+]]

unsigned bits(Record *p) {
  return __builtin_preserve_access_index(p->bits);
}
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0({{.*}}, i32 2, i32 4), {{.*}}!llvm.preserve.access.index ![[RECORD]]

long *member(Union *p) {
  return __builtin_preserve_access_index(&p->second);
}
// CHECK: call ptr @llvm.preserve.union.access.index.p0.p0({{.*}}, i32 2), {{.*}}!llvm.preserve.access.index ![[UNION:[0-9]+]]

int *dynamic(Dynamic *p) {
  return __builtin_preserve_access_index(&p->field);
}
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0({{.*}}, i32 1, i32 1), {{.*}}!llvm.preserve.access.index ![[DYNAMIC:[0-9]+]]

int lambda(int a, int b) {
  return [a, b] { return __builtin_preserve_access_index(b); }();
}
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0({{.*}}, i32 1, i32 1), {{.*}}!llvm.preserve.access.index ![[LAMBDA:[0-9]+]]

// The intrinsic indices refer to the complete DI element lists, including
// bases, static members, and the vtable pointer before the accessed fields.
// CHECK-DAG: ![[RECORD]] = distinct !DICompositeType({{.*}}name: "Record", {{.*}}elements: ![[RECORD_ELEMENTS:[0-9]+]]
// CHECK-DAG: ![[RECORD_ELEMENTS]] = !{!{{[0-9]+}}, !{{[0-9]+}}, ![[FIELD:[0-9]+]], !{{[0-9]+}}, ![[BITS:[0-9]+]], !{{[0-9]+}}}
// CHECK-DAG: ![[FIELD]] = !DIDerivedType(tag: DW_TAG_member, name: "field"
// CHECK-DAG: ![[BITS]] = !DIDerivedType(tag: DW_TAG_member, name: "bits"
// CHECK-DAG: ![[UNION]] = distinct !DICompositeType({{.*}}name: "Union", {{.*}}elements: ![[UNION_ELEMENTS:[0-9]+]]
// CHECK-DAG: ![[UNION_ELEMENTS]] = !{!{{[0-9]+}}, !{{[0-9]+}}, ![[SECOND:[0-9]+]]}
// CHECK-DAG: ![[SECOND]] = !DIDerivedType(tag: DW_TAG_member, name: "second"
// CHECK-DAG: ![[DYNAMIC]] = distinct !DICompositeType({{.*}}name: "Dynamic", {{.*}}elements: ![[DYNAMIC_ELEMENTS:[0-9]+]]
// CHECK-DAG: ![[DYNAMIC_ELEMENTS]] = !{!{{[0-9]+}}, ![[DYNAMIC_FIELD:[0-9]+]], !{{[0-9]+}}}
// CHECK-DAG: ![[DYNAMIC_FIELD]] = !DIDerivedType(tag: DW_TAG_member, name: "field"
// CHECK-DAG: ![[LAMBDA]] = distinct !DICompositeType({{.*}}elements: ![[LAMBDA_ELEMENTS:[0-9]+]]
// CHECK-DAG: ![[LAMBDA_ELEMENTS]] = !{!{{[0-9]+}}, ![[CAPTURE:[0-9]+]]}
// CHECK-DAG: ![[CAPTURE]] = !DIDerivedType(tag: DW_TAG_member, name: "b"
// CHECK-DAG: !DICompositeType({{.*}}name: "Base", {{.*}}elements: ![[BASE_ELEMENTS:[0-9]+]]
// CHECK-DAG: ![[BASE_ELEMENTS]] = !{![[BASE_FIELD:[0-9]+]], !{{[0-9]+}}}
// CHECK-DAG: ![[BASE_FIELD]] = !DIDerivedType(tag: DW_TAG_member, name: "base"
