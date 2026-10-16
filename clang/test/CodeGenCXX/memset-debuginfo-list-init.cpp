// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

void use(void *);

struct A {
  char buf[64];
};

// CHECK-LABEL: define {{.*}}void @_Z18test_cpp_aggregatev()
void test_cpp_aggregate() {
  // CHECK: call void @llvm.memset.{{.*}} !dbg ![[LOC_AGG:[0-9]+]]
  A a = { 0 };
  use(&a);
}

// CHECK-LABEL: define {{.*}}void @_Z14test_cpp_arrayv()
void test_cpp_array() {
  // CHECK: call void @llvm.memset.{{.*}} !dbg ![[LOC_ARR:[0-9]+]]
  char array[24] = { 0 };
  use(array);
}

struct Base {
  char buf[32];
};
struct Derived : Base {
  int x;
  Derived() : Base(), x(0) {}
};

void test_derived() {
  Derived d;
  use(&d);
}

// CHECK-LABEL: define {{.*}}void @_Z21test_cpp_empty_bracesv()
void test_cpp_empty_braces() {
  // CHECK: call void @llvm.memset.{{.*}} !dbg ![[LOC_EMPTY:[0-9]+]]
  char array[24]{};
  use(array);
}

// CHECK-LABEL: define {{.*}}void @_Z18test_cpp_array_newv()
void test_cpp_array_new() {
  // CHECK: call void @llvm.memset.{{.*}} !dbg ![[LOC_NEW:[0-9]+]]
  char *p = new char[24]{};
  use(p);
}

// CHECK-LABEL: define {{.*}}void @_ZN7DerivedC2Ev(
// CHECK: call void @llvm.memset.{{.*}} !dbg ![[LOC_BASE:[0-9]+]]

// CHECK-DAG: ![[LOC_AGG]] = !DILocation(line: 0, scope: ![[SP_MEMSET:[0-9]+]], inlinedAt: ![[INLINED_AGG:[0-9]+]])
// CHECK-DAG: ![[SP_MEMSET]] = distinct !DISubprogram(name: "memset", scope: ![[FILE:[0-9]+]], file: ![[FILE]], type: ![[SUBROUTINE_TYPE:[0-9]+]], flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: ![[CU:[0-9]+]])

// CHECK-DAG: ![[LOC_ARR]] = !DILocation(line: 0, scope: ![[SP_MEMSET]], inlinedAt: ![[INLINED_ARR:[0-9]+]])
// CHECK-DAG: ![[LOC_BASE]] = !DILocation(line: 0, scope: ![[SP_MEMSET]], inlinedAt: ![[INLINED_BASE:[0-9]+]])
// CHECK-DAG: ![[LOC_EMPTY]] = !DILocation(line: 0, scope: ![[SP_MEMSET]], inlinedAt: ![[INLINED_EMPTY:[0-9]+]])
// CHECK-DAG: ![[LOC_NEW]] = !DILocation(line: 0, scope: ![[SP_MEMSET]], inlinedAt: ![[INLINED_NEW:[0-9]+]])


