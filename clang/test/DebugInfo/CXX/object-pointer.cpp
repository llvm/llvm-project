// RUN: %clang_cc1 -x c++ -std=c++23 -debug-info-kind=limited -emit-llvm < %s | FileCheck %s

// CHECK: !DISubprogram(name: "bar",
// CHECK-SAME:          flags: DIFlagPrototyped
// CHECK: !DIDerivedType(tag: DW_TAG_pointer_type
// CHECK-SAME:           flags: DIFlagArtificial | DIFlagObjectPointer
//
// CHECK: !DISubprogram(name: "explicit_this",
//                      flags: DIFlagPrototyped
//
// CHECK: !DIDerivedType(tag: DW_TAG_rvalue_reference_type
// CHECK-SAME:           flags: DIFlagObjectPointer)
//
// CHECK: !DILocalVariable(name: "this", arg: 1
// CHECK-SAME:             flags: DIFlagArtificial | DIFlagObjectPointer
//
// CHECK-NOT: DIFlagArtificial
// CHECK: !DILocalVariable(arg: 1, {{.*}}, flags: DIFlagObjectPointer)

struct Foo {
  void bar() {}
  void explicit_this(this Foo &&) {}
};

void f() {
  Foo{}.bar();
  Foo{}.explicit_this();
}
