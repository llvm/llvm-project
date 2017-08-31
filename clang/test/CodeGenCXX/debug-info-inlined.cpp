// RUN: %clang_cc1 -emit-llvm -triple i686-pc-windows-msvc19.0.24213 -gcodeview -debug-info-kind=limited -std=c++14 %s -o - | FileCheck %s
// PR33997.
struct WithDtor {
  ~WithDtor();
};
struct Base {
  Base(WithDtor);
};
class Forward : Base {
  using Base::Base;
};
class A : Forward {
  A();
};
class B : Forward {
  B();
};
A::A() : Forward(WithDtor()) {}

B::B() : Forward(WithDtor()) {}

// CHECK: define{{.*}}A
// CHECK-NOT: {{ ret }}
// CHECK: store %class.Forward* %
// CHECK-SAME: %class.Forward** %
// CHECK-SAME: !dbg ![[INL:[0-9]+]]

// CHECK: ![[INL]] = !DILocation(line: 10, scope: ![[SP:[0-9]+]], inlinedAt:
// CHECK: ![[SP]] = distinct !DISubprogram(name: "Base", {{.*}}isDefinition: true
