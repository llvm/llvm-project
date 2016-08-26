// RUN: %clang_cc1 -triple x86_64-windows-gnu %s -emit-llvm -o - | FileCheck %s
struct A { int a; };
struct B : virtual A { int b; };
B b;

// CHECK: @_ZTI1B = linkonce_odr constant { i8*, i8*, i32, i32, i8*, i64 }
// CHECK-SAME:  i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2) to i8*),
// CHECK-SAME:  i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1B, i32 0, i32 0),
// CHECK-SAME:  i32 0,
// CHECK-SAME:  i32 1,
// CHECK-SAME:  i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*),
//    This i64 is important, it should be an i64, not an i32.
// CHECK-SAME:  i64 -6141 }, comdat
