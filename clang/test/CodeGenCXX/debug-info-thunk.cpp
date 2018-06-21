// RUN: %clang_cc1 %s -triple %itanium_abi_triple -debug-info-kind=limited -S -emit-llvm -o - | FileCheck %s

struct A {
  virtual void f();
};

struct B {
  virtual void f();
};

struct C : A, B {
  virtual void f();
};

void C::f() { }
// CHECK: define {{.*}}void @_ZThn{{[48]}}_N1C1fEv
// CHECK-SAME: !dbg ![[SP:[0-9]+]]
// CHECK-NOT: {{ret }}
// CHECK: = load{{.*}} !dbg ![[DBG:[0-9]+]]
// CHECK-NOT: {{ret }}
// CHECK: ret void, !dbg ![[DBG]]
//
// CHECK: ![[SP]] = distinct !DISubprogram(linkageName: "_ZThn{{[48]}}_N1C1fEv"
// CHECK-SAME:          line: 15
// CHECK-SAME:          isDefinition: true
// CHECK-SAME:          DIFlagArtificial
// CHECK-SAME:          ){{$}}
//
// CHECK: ![[DBG]] = !DILocation(line: 0
