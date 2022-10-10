//  Test for debug info for C++11 auto return member functions
// RUN: %clang_cc1 -dwarf-version=5  -emit-llvm -triple x86_64-linux-gnu %s -o - \
// RUN:   -O0 -disable-llvm-passes \
// RUN:   -debug-info-kind=standalone \
// RUN: | FileCheck --implicit-check-not="\"auto\"" --implicit-check-not=DISubprogram %s

// CHECK: !DISubprogram(name: "findMax",{{.*}}, type: [[FUN_TYPE:![0-9]+]], {{.*}}spFlags: DISPFlagDefinition, {{.*}} declaration: [[DECL:![0-9]+]]

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "myClass",
// CHECK-SAME: elements: [[MEMBERS:![0-9]+]],

// CHECK: [[MEMBERS]] = !{}

// CHECK: [[FUN_TYPE]] = !DISubroutineType(types: [[TYPE_NODE:![0-9]+]])
// CHECK-NEXT: [[TYPE_NODE]] = !{[[DOUBLE_TYPE:![0-9]+]],
// CHECK-NEXT: [[DOUBLE_TYPE]] = !DIBasicType(name: "double",
// CHECK: [[DECL]] = !DISubprogram(name: "findMax",{{.*}}, type: [[FUN_TYPE]],

struct myClass {
  auto findMax();
};

auto myClass::findMax() {
  return 0.0;
}
