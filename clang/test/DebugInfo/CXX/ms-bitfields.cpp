// RUN: %clang_cc1 -triple x86_64-pc-win32 -debug-info-kind=limited -gcodeview %s -emit-llvm -o - | FileCheck %s

#pragma pack(1)
struct S {
  char : 8;
  short   : 8;
  short x : 8;
} s;

// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "x", {{.*}}, size: 8, offset: 16, flags: DIFlagBitField, extraData: i64 8)
