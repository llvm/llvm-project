// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -debug-info-kind=line-tables-only %s -o - | FileCheck %s
// Crasher for PR22929.
class Base {
  virtual void VariadicFunction(...);
};

class Derived : public virtual Base {
  virtual void VariadicFunction(...);
};

void Derived::VariadicFunction(...) { }

// CHECK: define void @_ZN7Derived16VariadicFunctionEz({{.*}} !dbg ![[SP:[0-9]+]]
// CHECK: ret void, !dbg ![[LOC:[0-9]+]]
// CHECK-LABEL: define void @_ZT{{.+}}N7Derived16VariadicFunctionEz(
// CHECK: ret void, !dbg ![[LOC:[0-9]+]]
//
// CHECK: !llvm.dbg.cu = !{![[CU:[0-9]+]]}
//
// CHECK: ![[CU]] = distinct !DICompileUnit({{.*}} subprograms: ![[SPs:[0-9]+]]
// CHECK: ![[SPs]] = !{![[SP]]}
// CHECK: ![[SP]] = distinct !DISubprogram(name: "VariadicFunction"
// CHECK: ![[LOC]] = !DILocation({{.*}}scope: ![[SP]])
