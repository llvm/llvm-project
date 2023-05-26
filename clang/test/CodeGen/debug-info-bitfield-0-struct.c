// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=limited %s | FileCheck --check-prefixes NOSEPARATOR,BOTH %s
// RUN: %clang_cc1 -triple amdgcn-unk-unk -o - -emit-llvm -debug-info-kind=limited %s | FileCheck --check-prefixes SEPARATOR,BOTH %s

struct First {
  // BOTH-DAG: ![[FIRST:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "First", file: !{{[0-9]+}}, line: {{[0-9]+}}, size: 32, elements: ![[FIRST_ELEMENTS:[0-9]+]])
  // BOTH-DAG: ![[FIRST_ELEMENTS]] = !{![[FIRST_X:[0-9]+]], ![[FIRST_Y:[0-9]+]]}
  // BOTH-DAG: ![[FIRST_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: ![[FIRST]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, flags: DIFlagBitField, extraData: i64 0)
  // BOTH-DAG: ![[FIRST_Y]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: ![[FIRST]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, offset: 4, flags: DIFlagBitField, extraData: i64 0)
  int : 0;
  int x : 4;
  int y : 4;
};

struct FirstDuplicate {
  // BOTH-DAG: ![[FIRSTDUP:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "FirstDuplicate", file: !{{[0-9]+}}, line: {{[0-9]+}}, size: 32, elements: ![[FIRSTDUP_ELEMENTS:[0-9]+]])
  // BOTH-DAG: ![[FIRSTDUP_ELEMENTS]] = !{![[FIRSTDUP_X:[0-9]+]], ![[FIRSTDUP_Y:[0-9]+]]}
  // BOTH-DAG: ![[FIRSTDUP_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: ![[FIRSTDUP]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, flags: DIFlagBitField, extraData: i64 0)
  // BOTH-DAG: ![[FIRSTDUP_Y]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: ![[FIRSTDUP]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, offset: 4, flags: DIFlagBitField, extraData: i64 0)
  int : 0;
  int : 0;
  int x : 4;
  int y : 4;
};

struct Second {
  // BOTH-DAG: ![[SECOND:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Second", file: !{{[0-9]+}}, line: {{[0-9]+}}, size: 64, elements: ![[SECOND_ELEMENTS:[0-9]+]])

  // NOSEPARATOR-DAG: ![[SECOND_ELEMENTS]] = !{![[SECOND_X:[0-9]+]], ![[SECOND_Y:[0-9]+]]}
  // SEPARATOR-DAG: ![[SECOND_ELEMENTS]] = !{![[SECOND_X:[0-9]+]], ![[SECOND_ZERO:[0-9]+]], ![[SECOND_Y:[0-9]+]]}

  // BOTH-DAG: ![[SECOND_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: ![[SECOND]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, flags: DIFlagBitField, extraData: i64 0)
  // SEPARATOR-DAG: ![[SECOND_ZERO]] = !DIDerivedType(tag: DW_TAG_member, scope: ![[SECOND]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, offset: 32, flags: DIFlagBitField, extraData: i64 32)
  // BOTH-DAG: ![[SECOND_Y]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: ![[SECOND]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, offset: 32, flags: DIFlagBitField, extraData: i64 32)
  int x : 4;
  int : 0;
  int y : 4;
};

struct SecondDuplicate {
  // BOTH-DAG: ![[SECONDDUP:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "SecondDuplicate", file: !{{[0-9]+}}, line: {{[0-9]+}}, size: 64, elements: ![[SECONDDUP_ELEMENTS:[0-9]+]])

  // NOSEPARATOR-DAG: ![[SECONDDUP_ELEMENTS]] = !{![[SECONDDUP_X:[0-9]+]], ![[SECONDDUP_Y:[0-9]+]]}
  // SEPARATOR-DAG: ![[SECONDDUP_ELEMENTS]] = !{![[SECONDDUP_X:[0-9]+]], ![[SECONDDUP_ZERO:[0-9]+]], ![[SECONDDUP_Y:[0-9]+]]}

  // BOTH-DAG: ![[SECONDDUP_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: ![[SECONDDUP]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, flags: DIFlagBitField, extraData: i64 0)
  // SEPARATOR-DAG: ![[SECONDDUP_ZERO]] = !DIDerivedType(tag: DW_TAG_member, scope: ![[SECONDDUP]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, offset: 32, flags: DIFlagBitField, extraData: i64 32)
  // BOTH-DAG: ![[SECONDDUP_Y]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: ![[SECONDDUP]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, offset: 32, flags: DIFlagBitField, extraData: i64 32)
  int x : 4;
  int : 0;
  int : 0;
  int y : 4;
};

struct Last {
  // BOTH-DAG: ![[LAST:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Last", file: !{{[0-9]+}}, line: {{[0-9]+}}, size: 32, elements: ![[LAST_ELEMENTS:[0-9]+]])
  // BOTH-DAG: ![[LAST_ELEMENTS]] = !{![[LAST_X:[0-9]+]], ![[LAST_Y:[0-9]+]]}
  // BOTH-DAG: ![[LAST_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: ![[LAST]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, flags: DIFlagBitField, extraData: i64 0)
  // BOTH-DAG: ![[LAST_Y]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: ![[LAST]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, offset: 4, flags: DIFlagBitField, extraData: i64 0)
  int x : 4;
  int y : 4;
  int : 0;
};

struct Several {
  // BOTH-DAG: ![[SEVERAL:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Several", file: !{{[0-9]+}}, line: {{[0-9]+}}, size: 96, elements: ![[SEVERAL_ELEMENTS:[0-9]+]])

  // SEPARATOR-DAG: ![[SEVERAL_ELEMENTS]] = !{![[SEVERAL_X:[0-9]+]], ![[SEVERAL_FIRST_ZERO:[0-9]+]], ![[SEVERAL_Y:[0-9]+]], ![[SEVERAL_SECOND_ZERO:[0-9]+]], ![[SEVERAL_Z:[0-9]+]]}
  // NOSEPARATOR-DAG: ![[SEVERAL_ELEMENTS]] = !{![[SEVERAL_X:[0-9]+]], ![[SEVERAL_Y:[0-9]+]], ![[SEVERAL_Z:[0-9]+]]}

  // BOTH-DAG: ![[SEVERAL_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: ![[SEVERAL]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, flags: DIFlagBitField, extraData: i64 0)
  // SEPARATOR-DAG: ![[SEVERAL_FIRST_ZERO]] = !DIDerivedType(tag: DW_TAG_member, scope: ![[SEVERAL]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, offset: 32, flags: DIFlagBitField, extraData: i64 32)
  // BOTH-DAG: ![[SEVERAL_Y]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: ![[SEVERAL]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, offset: 32, flags: DIFlagBitField, extraData: i64 32)
  // SEPARATOR-DAG: ![[SEVERAL_SECOND_ZERO]] = !DIDerivedType(tag: DW_TAG_member, scope: ![[SEVERAL]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, offset: 64, flags: DIFlagBitField, extraData: i64 64)
  // BOTH-DAG: ![[SEVERAL_Z]] = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: ![[SEVERAL]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, offset: 64, flags: DIFlagBitField, extraData: i64 64)
  int x : 4;
  int : 0;
  int y : 4;
  int : 0;
  int z : 4;
};

struct None_A {
  // BOTH-DAG: ![[NONE_A:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "None_A", file: !{{[0-9]+}}, line: {{[0-9]+}}, size: 64, elements: ![[NONE_A_ELEMENTS:[0-9]+]])
  // BOTH-DAG: ![[NONE_A_ELEMENTS]] = !{![[NONE_A_FIELD:[0-9]+]], ![[NONE_A_X:[0-9]+]]}
  // BOTH-DAG: ![[NONE_A_FIELD]] = !DIDerivedType(tag: DW_TAG_member, name: "field", scope: ![[NONE_A]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 32)
  // BOTH-DAG: ![[NONE_A_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: ![[NONE_A]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, offset: 32, flags: DIFlagBitField, extraData: i64 32)
  int : 0;
  int field;
  int x : 4;
};

struct None_B {
  // BOTH-DAG: ![[NONE_B:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "None_B", file: !{{[0-9]+}}, line: {{[0-9]+}}, size: 64, elements: ![[NONE_B_ELEMENTS:[0-9]+]])
  // BOTH-DAG: ![[NONE_B_ELEMENTS]] = !{![[NONE_B_FIELD:[0-9]+]], ![[NONE_B_X:[0-9]+]], ![[NONE_B_Y:[0-9]+]]}
  // BOTH-DAG: ![[NONE_B_FIELD]] = !DIDerivedType(tag: DW_TAG_member, name: "field", scope: ![[NONE_B]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 32)
  // BOTH-DAG: ![[NONE_B_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: ![[NONE_B]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, offset: 32, flags: DIFlagBitField, extraData: i64 32)
  // BOTH-DAG: ![[NONE_B_Y]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: ![[NONE_B]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 4, offset: 36, flags: DIFlagBitField, extraData: i64 32)
  int field;
  int : 0;
  int x : 4;
  int y : 4;
};

struct None_C {
  // BOTH-DAG: ![[NONE_C:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "None_C", file: !{{[0-9]+}}, line: {{[0-9]+}}, size: 32, elements: ![[NONE_C_ELEMENTS:[0-9]+]])
  // BOTH-DAG: ![[NONE_C_ELEMENTS]] = !{![[NONE_C_X:[0-9]+]], ![[NONE_C_Y:[0-9]+]], ![[NONE_C_A:[0-9]+]], ![[NONE_C_B:[0-9]+]]}
  // BOTH-DAG: ![[NONE_C_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: ![[NONE_C]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 8, flags: DIFlagBitField, extraData: i64 0)
  // BOTH-DAG: ![[NONE_C_Y]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: ![[NONE_C]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 8, offset: 8, flags: DIFlagBitField, extraData: i64 0)
  // BOTH-DAG: ![[NONE_C_A]] = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: ![[NONE_C]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 8, offset: 16, flags: DIFlagBitField, extraData: i64 0)
  // BOTH-DAG: ![[NONE_C_B]] = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: ![[NONE_C]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 8, offset: 24, flags: DIFlagBitField, extraData: i64 0)
  char x : 8;
  char y : 8;
  char a : 8;
  char b : 8;
};

void foo(struct First f, struct FirstDuplicate fs, struct Second s, struct SecondDuplicate sd, struct Last l, struct Several ss, struct None_A na, struct None_B nb, struct None_C nc) {
  return;
}
