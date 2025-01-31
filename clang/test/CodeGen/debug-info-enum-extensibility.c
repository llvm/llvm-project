// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK-NOT:  enumKind
// CHECK:      !DICompositeType(tag: DW_TAG_enumeration_type, name: "ClosedEnum"
// CHECK-SAME:                  enumKind: 0)
// CHECK:      !DICompositeType(tag: DW_TAG_enumeration_type, name: "OpenEnum"
// CHECK-SAME:                  enumKind: 1)
// CHECK:      !DICompositeType(tag: DW_TAG_enumeration_type, name: "ClosedFlagEnum"
// CHECK-SAME:                  enumKind: 0)
// CHECK:      !DICompositeType(tag: DW_TAG_enumeration_type, name: "OpenFlagEnum"
// CHECK-SAME:                  enumKind: 1)
// CHECK:      !DICompositeType(tag: DW_TAG_enumeration_type, name: "MixedEnum"
// CHECK-SAME:                  enumKind: 1)

enum Enum {
  E0, E1
};

enum FlagEnum {
  FE0 = 1 << 0, FE1 = 1 << 1
};

enum __attribute__((enum_extensibility(closed))) ClosedEnum {
  A0, A1
};

enum __attribute__((enum_extensibility(open))) OpenEnum {
  B0, B1
};

enum __attribute__((enum_extensibility(closed),flag_enum)) ClosedFlagEnum {
  C0 = 1 << 0, C1 = 1 << 1
};

enum __attribute__((enum_extensibility(open),flag_enum)) OpenFlagEnum {
  D0 = 1 << 0, D1 = 1 << 1
};

enum __attribute__((enum_extensibility(open), enum_extensibility(closed))) MixedEnum {
  M0, M1
};

enum Enum e;
enum FlagEnum fe;
enum ClosedEnum ce;
enum OpenEnum oe;
enum ClosedFlagEnum cfe;
enum OpenFlagEnum ofe;
enum MixedEnum me;
