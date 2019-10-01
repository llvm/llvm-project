// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin -std=c++11 %s -o - | FileCheck %s

enum class A { A1=1 };                 // underlying type is int by default
enum class B: unsigned long { B1=1 };  // underlying type is unsigned long
enum C { C1 = 1 };
enum D : short; // enum forward declaration
enum Z : int;
A a;
B b;
C c;
D d;

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "A"
// CHECK-SAME:             line: 3
// CHECK-SAME:             baseType: ![[INT:[0-9]+]]
// CHECK-SAME:             size: 32
// CHECK-NOT:              offset:
// CHECK-SAME:             flags: DIFlagEnumClass
// CHECK-SAME:             ){{$}}
// CHECK: ![[INT]] = !DIBasicType(name: "int"
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "B"
// CHECK-SAME:             line: 4
// CHECK-SAME:             baseType: ![[ULONG:[0-9]+]]
// CHECK-SAME:             size: 64
// CHECK-NOT:              offset:
// CHECK-SAME:             flags: DIFlagEnumClass
// CHECK-SAME:             ){{$}}
// CHECK: ![[ULONG]] = !DIBasicType(name: "long unsigned int"
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "C"
// CHECK-SAME:             line: 5
// CHECK-SAME:             baseType: ![[ULONG:[0-9]+]]
// CHECK-SAME:             size: 32
// CHECK-NOT:              offset:
// CHECK-NOT:              flags:
// CHECK-SAME:             ){{$}}

namespace PR14029 {
  // Make sure this doesn't crash/assert.
  template <typename T> struct Test {
    enum class Tag {
      test = 0
    };
    Test() {
      auto t = Tag::test;
    }
    Tag tag() const { return static_cast<Tag>(1); }
  };
  Test<int> t;
}

namespace test2 {
// FIXME: this should just be a declaration under -fno-standalone-debug
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E"
// CHECK-SAME:             scope: [[TEST2:![0-9]+]]
// CHECK-NOT:              DIFlagEnumClass
// CHECK-SAME:             elements: [[TEST_ENUMS:![0-9]+]]
// CHECK-SAME:             identifier: "_ZTSN5test21EE"
// CHECK: [[TEST2]] = !DINamespace(name: "test2"
// CHECK: [[TEST_ENUMS]] = !{[[TEST_E:![0-9]*]]}
// CHECK: [[TEST_E]] = !DIEnumerator(name: "e", value: 0)
enum E : int;
void func(E *) {
}
enum E : int { e };
}

namespace test3 {
// FIXME: this should just be a declaration under -fno-standalone-debug
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E"
// CHECK-SAME:             scope: [[TEST3:![0-9]+]]
// CHECK-NOT:              DIFlagEnumClass
// CHECK-SAME:             elements: [[TEST_ENUMS]]
// CHECK-SAME:             identifier: "_ZTSN5test31EE"
// CHECK: [[TEST3]] = !DINamespace(name: "test3"
enum E : int { e };
void func(E *) {
}
}

namespace test4 {
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E"
// CHECK-SAME:             scope: [[TEST4:![0-9]+]]
// CHECK-NOT:              DIFlagEnumClass
// CHECK-SAME:             elements: [[TEST_ENUMS]]
// CHECK-SAME:             identifier: "_ZTSN5test41EE"
// CHECK: [[TEST4]] = !DINamespace(name: "test4"
enum E : int;
void f1(E *) {
}
enum E : int { e };
void f2(E) {
}
}

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "D"
// CHECK-SAME:             line: 6
// CHECK-SAME:             size: 16
// CHECK-NOT:              offset:
// CHECK-SAME:             flags: DIFlagFwdDecl

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "Z"
// CHECK-NOT:              scope:
// CHECK-SAME:             flags: DIFlagFwdDecl
void fz() { Z z; }

namespace test5 {
// CHECK: [[TEST5:![0-9]+]] = !DINamespace(name: "test5"
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E"
// CHECK-SAME:             scope: [[TEST5]]
// CHECK-SAME:             flags: DIFlagFwdDecl
// CHECK-SAME:             identifier: "_ZTSN5test51EE"
enum E : int;
void f1(E *) {
}
}

namespace test6 {
// Ensure typedef'd enums aren't manifest by debug info generation.
// This could cause "typedef changes linkage of anonymous type, but linkage was
// already computed" errors.
// CHECK-NOT: test6
typedef enum {
} E;
}
