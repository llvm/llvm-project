// Test enumeration representation in debuig info metadata:
// * test value representation for each possible underlying integer type
// * test the integer type is as expected
// * test the DW_AT_enum_class attribute is present (resp. absent) as expected.

// RUN: %clang -target x86_64-linux -g -S -emit-llvm -o - %s | FileCheck %s


enum class E0 : signed char {
  A0 = -128,
  B0 = 127,
} x0;
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E0"
// CHECK-SAME: baseType: ![[SCHAR:[0-9]+]]
// CHECK-SAME: DIFlagEnumClass
// CHECK-SAME: elements: ![[ELTS0:[0-9]+]]
// CHECK: ![[SCHAR]] = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
// CHECK: ![[ELTS0]] = !{![[A0:[0-9]+]], ![[B0:[0-9]+]]}
// CHECK: ![[A0]] = !DIEnumerator(name: "A0", value: -128)
// CHECK: ![[B0]] = !DIEnumerator(name: "B0", value: 127)

enum class E1 : unsigned char { A1 = 255 } x1;
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E1"
// CHECK-SAME: baseType: ![[UCHAR:[0-9]+]]
// CHECK-SAME: DIFlagEnumClass
// CHECK-SAME: elements: ![[ELTS1:[0-9]+]]
// CHECK: ![[UCHAR]] = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
// CHECK: ![[ELTS1]] = !{![[A1:[0-9]+]]}
// CHECK: ![[A1]] = !DIEnumerator(name: "A1", value: 255, isUnsigned: true)

enum class E2 : signed short {
  A2 = -32768,
  B2 = 32767,
} x2;
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E2"
// CHECK-SAME: baseType: ![[SHORT:[0-9]+]]
// CHECK-SAME: DIFlagEnumClass
// CHECK-SAME: elements: ![[ELTS2:[0-9]+]]
// CHECK: ![[SHORT]] = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
// CHECK: ![[ELTS2]] = !{![[A2:[0-9]+]], ![[B2:[0-9]+]]}
// CHECK: ![[A2]] = !DIEnumerator(name: "A2", value: -32768)
// CHECK: ![[B2]] = !DIEnumerator(name: "B2", value: 32767)

enum class E3 : unsigned short { A3 = 65535 } x3;
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E3"
// CHECK-SAME: baseType: ![[USHORT:[0-9]+]]
// CHECK-SAME: DIFlagEnumClass
// CHECK-SAME: elements: ![[ELTS3:[0-9]+]]
// CHECK: ![[USHORT]] = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
// CHECK: ![[ELTS3]] = !{![[A3:[0-9]+]]}
// CHECK: ![[A3]] = !DIEnumerator(name: "A3", value: 65535, isUnsigned: true)

enum class E4 : signed int { A4 = -2147483648, B4 = 2147483647 } x4;
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E4"
// CHECK-SAME: baseType: ![[INT:[0-9]+]]
// CHECK-SAME: DIFlagEnumClass
// CHECK-SAME: elements: ![[ELTS4:[0-9]+]]
// CHECK: ![[INT]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK: ![[ELTS4]] = !{![[A4:[0-9]+]], ![[B4:[0-9]+]]}
// CHECK: ![[A4]] = !DIEnumerator(name: "A4", value: -2147483648)
// CHECK: ![[B4]] = !DIEnumerator(name: "B4", value: 2147483647)

enum class E5 : unsigned int { A5 = 4294967295 } x5;
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E5"
// CHECK-SAME: baseType: ![[UINT:[0-9]+]]
// CHECK-SAME: DIFlagEnumClass
// CHECK-SAME: elements: ![[ELTS5:[0-9]+]]
// CHECK: ![[UINT]] = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
// CHECK: ![[ELTS5]] = !{![[A5:[0-9]+]]}
// CHECK: ![[A5]] = !DIEnumerator(name: "A5", value: 4294967295, isUnsigned: true)

enum class E6 : signed long long {
  A6 = -9223372036854775807LL - 1,
  B6 = 9223372036854775807LL
} x6;
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E6"
// CHECK-SAME: baseType: ![[LONG:[0-9]+]]
// CHECK-SAME: DIFlagEnumClass
// CHECK-SAME: elements: ![[ELTS6:[0-9]+]]
// CHECK: ![[LONG]] = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
// CHECK: ![[ELTS6]] = !{![[A6:[0-9]+]], ![[B6:[0-9]+]]}
// CHECK: ![[A6]] = !DIEnumerator(name: "A6", value: -9223372036854775808)
// CHECK: ![[B6]] = !DIEnumerator(name: "B6", value: 9223372036854775807)

enum class E7 : unsigned long long { A7 = 18446744073709551615ULL } x7;
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E7"
// CHECK-SAME: baseType: ![[ULONG:[0-9]+]]
// CHECK-SAME: DIFlagEnumClass
// CHECK-SAME: elements: ![[ELTS7:[0-9]+]]
// CHECK: ![[ULONG]] = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)
// CHECK: ![[ELTS7]] = !{![[A7:[0-9]+]]}
// CHECK: ![[A7]] = !DIEnumerator(name: "A7", value: 18446744073709551615, isUnsigned: true)

// Also test the FixedEnum flag is not present for old-style enumerations.
enum E8 { A8 = -128, B8 = 127 } x8;
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E8"
// CHECK-SAME: baseType: ![[INT]]
// CHECK-NOT: DIFlagEnumClass
// CHECK: !DIEnumerator(name: "A8", value: -128)

