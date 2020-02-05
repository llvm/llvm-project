//  Test for debug info for C++11 auto return member functions
// RUN: %clang_cc1 -dwarf-version=5  -emit-llvm -triple x86_64-linux-gnu %s -o - \
// RUN:   -O0 -disable-llvm-passes \
// RUN:   -debug-info-kind=standalone \
// RUN: | FileCheck %s

// CHECK: !DISubprogram(name: "findMax",{{.*}}, type: ![[FUN_TYPE:[0-9]+]],{{.*}}

// CHECK: ![[FUN_TYPE]] = !DISubroutineType(types: ![[TYPE_NODE:[0-9]+]])
// CHECK-NEXT: ![[TYPE_NODE]] = !{![[DOUBLE_TYPE:[0-9]+]], {{.*}}
// CHECK-NEXT: ![[DOUBLE_TYPE]] = !DIBasicType(name: "double", {{.*}})

// CHECK: !DISubroutineType(types: ![[TYPE_DECL_NODE:[0-9]+]])
// CHECK-NEXT: ![[TYPE_DECL_NODE]] = !{![[AUTO_TYPE:[0-9]+]], {{.*}}
// CHECK-NEXT: ![[AUTO_TYPE]] = !DIBasicType(tag: DW_TAG_unspecified_type, name: "auto")
struct myClass {
  auto findMax();
};

auto myClass::findMax() {
  return 0.0;
}
