// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-linux-gnu %s -o - | FileCheck %s

// Make sure that we emit a global variable for each of the members of the
// anonymous union.

static union {
  int c;
  int d;
  union {
    int a;
  };
  struct {
    int b;
  };
};

int test_it() {
  c = 1;
  d = 2;
  a = 4;
  return (c == 1);
}

void foo() {
  union {
    int i;
    char c;
  };
  i = 8;
}

// A funky reinterpret cast idiom that we used to crash on.
template <class T>
unsigned char *buildBytes(const T v) {
  static union {
    unsigned char result[sizeof(T)];
    T value;
  };
  value = v;
  return result;
}

void instantiate(int x) {
  buildBytes(x);
}

// CHECK: !DIGlobalVariable(name: "c",{{.*}} file: [[FILE:.*]], line: 6,{{.*}} isLocal: true, isDefinition: true
// CHECK: !DIGlobalVariable(name: "d",{{.*}} file: [[FILE]], line: 6,{{.*}} isLocal: true, isDefinition: true
// CHECK: [[FILE]] = !DIFile(filename: "{{.*}}debug-info-anon-union-vars.cpp",
// CHECK: !DIGlobalVariable(name: "a",{{.*}} file: [[FILE]], line: 6,{{.*}} isLocal: true, isDefinition: true
// CHECK: !DIGlobalVariable(name: "b",{{.*}} file: [[FILE]], line: 6,{{.*}} isLocal: true, isDefinition: true
// CHECK: !DIGlobalVariable(name: "result", {{.*}} isLocal: false, isDefinition: true
// CHECK: !DIGlobalVariable(name: "value", {{.*}} isLocal: false, isDefinition: true
// CHECK: !DILocalVariable(name: "i", {{.*}}, flags: DIFlagArtificial
// CHECK: !DILocalVariable(name: "c", {{.*}}, flags: DIFlagArtificial
// CHECK: !DILocalVariable(
// CHECK-NOT: name:
// CHECK: type: ![[UNION:[0-9]+]]
// CHECK: ![[UNION]] = distinct !DICompositeType(tag: DW_TAG_union_type,
// CHECK-NOT: name:
// CHECK: elements
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "i", scope: ![[UNION]],
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "c", scope: ![[UNION]],
