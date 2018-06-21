// RUN: %clang_cc1 -emit-llvm -triple=i386-pc-win32 -fms-compatibility %s -o - | FileCheck %s

struct S {
  static const int NoInit_Ref;
  static const int Inline_NotDef_NotRef = 5;
  static const int Inline_NotDef_Ref = 5;
  static const int Inline_Def_NotRef = 5;
  static const int Inline_Def_Ref = 5;
  static const int OutOfLine_Def_NotRef;
  static const int OutOfLine_Def_Ref;
};

const int *foo1() {
  return &S::NoInit_Ref;
};

const int *foo2() {
  return &S::Inline_NotDef_Ref;
};

const int *foo3() {
  return &S::Inline_Def_Ref;
};

const int *foo4() {
    return &S::OutOfLine_Def_Ref;
};

const int S::Inline_Def_NotRef;
const int S::Inline_Def_Ref;
const int S::OutOfLine_Def_NotRef = 5;
const int S::OutOfLine_Def_Ref = 5;


// No initialization.
// CHECK-DAG: @"\01?NoInit_Ref@S@@2HB" = external constant i32

// Inline initialization, no real definiton, not referenced.
// CHECK-NOT: @"\01?Inline_NotDef_NotRef@S@@2HB" = {{.*}} constant i32 5

// Inline initialization, no real definiton, referenced.
// CHECK-DAG: @"\01?Inline_NotDef_Ref@S@@2HB" = linkonce_odr constant i32 5, comdat, align 4

// Inline initialization, real definiton, not referenced.
// CHECK-NOT: @"\01?Inline_Def_NotRef@S@@2HB" = constant i32 5, align 4

// Inline initialization, real definiton, referenced.
// CHECK-DAG: @"\01?Inline_Def_Ref@S@@2HB" = linkonce_odr constant i32 5, comdat, align 4

// Out-of-line initialization.
// CHECK-DAG: @"\01?OutOfLine_Def_NotRef@S@@2HB" = constant i32 5, align 4
// CHECK-DAG: @"\01?OutOfLine_Def_Ref@S@@2HB" = constant i32 5, align 4
