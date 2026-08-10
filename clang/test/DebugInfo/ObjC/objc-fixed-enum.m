// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s
// The DWARF standard says the underlying data type of an enum may be
// stored in an DW_AT_type entry in the enum DIE. This is useful to have
// so the debugger knows about the signedness of the underlying type.

typedef long NSInteger;
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type

// Enum with no specified underlying type
typedef enum {
  Enum0One,
  Enum0Two
} Enum0;

// Enum declared with the NS_ENUM macro
typedef NS_ENUM(NSInteger, Enum1) {
  Enum1One = -1,
  Enum1Two
};

// Enum declared with a fixed underlying type
typedef enum : NSInteger {
  Enum2One = -1,
  Enum2Two
} Enum2;

// Typedef and declaration separately
enum : NSInteger
{
  Enum3One = -1,
  Enum3Two
};
typedef NSInteger Enum3;

int main(void) {
  Enum0 e0 = Enum0One;
  // CHECK: #dbg_declare({{.*}}, ![[ENUM0:[0-9]+]], !{{.*}})
  Enum1 e1 = Enum1One;
  // CHECK: #dbg_declare({{.*}}, ![[ENUM1:[0-9]+]], !{{.*}})
  Enum2 e2 = Enum2One;
  // CHECK: #dbg_declare({{.*}}, ![[ENUM2:[0-9]+]], !{{.*}})
  Enum3 e3 = Enum3One;
  // CHECK: #dbg_declare({{.*}}, ![[ENUM3:[0-9]+]], !{{.*}})

  // -Werror and the following line ensures that these enums are not
  // -treated as C++11 strongly typed enums.
  return e0 != e1 && e1 == e2 && e2 == e3;
}
// CHECK: ![[ENUMERATOR0:[0-9]+]] = !DICompositeType(tag: DW_TAG_enumeration_type
// CHECK-SAME:                                       line: 10,
// CHECK: ![[ENUMERATOR1:[0-9]+]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Enum1"
// CHECK-SAME:                                       line: 16
// CHECK-SAME:                                       baseType: ![[ENUMERATOR3:[0-9]+]]
// CHECK: ![[ENUMERATOR3]] = !DIDerivedType(tag: DW_TAG_typedef, name: "NSInteger"
// CHECK-SAME:                              line: 6
// CHECK-SAME:                              baseType: ![[LONGINT:[0-9]+]]
// CHECK: ![[LONGINT]] = !DIBasicType(name: "long"
// CHECK: ![[ENUMERATOR2:[0-9]+]] = !DICompositeType(tag: DW_TAG_enumeration_type,
// CHECK-SAME:                                       line: 22
// CHECK-SAME:                                       baseType: ![[ENUMERATOR3]]

// CHECK: ![[ENUM0]] = !DILocalVariable(name: "e0"
// CHECK-SAME:                          type: ![[TYPE0:[0-9]+]]
// CHECK: ![[TYPE0]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Enum0",
// CHECK-SAME:                        baseType: ![[ENUMERATOR0]]

// CHECK: ![[ENUM1]] = !DILocalVariable(name: "e1"
// CHECK-SAME:                          type: ![[TYPE1:[0-9]+]]
// CHECK: ![[TYPE1]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Enum1"
// CHECK-SAME:                        baseType: ![[ENUMERATOR1]]

// CHECK: ![[ENUM2]] = !DILocalVariable(name: "e2"
// CHECK-SAME:                          type: ![[TYPE2:[0-9]+]]
// CHECK: ![[TYPE2]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Enum2"
// CHECK-SAME:                        baseType: ![[ENUMERATOR2]]

// CHECK: ![[ENUM3]] = !DILocalVariable(name: "e3"
// CHECK-SAME:                          type: ![[TYPE3:[0-9]+]]
// CHECK: ![[TYPE3]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Enum3"
// CHECK-SAME:                        baseType: ![[ENUMERATOR3]]
