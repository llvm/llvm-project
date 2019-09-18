// RUN: %clang_cc1 %s -I%S -triple=arm64-apple-ios -fptrauth-calls -std=c++11 -emit-llvm -o - | FileCheck %s
#include <typeinfo>

struct A { int a; };

// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTS1A = linkonce_odr hidden constant [3 x i8] c"1A\00"
// CHECK: @_ZTI1A = linkonce_odr hidden constant { i8*, i8* } { i8* bitcast ({ i8*, i32, i64, i64 }* @_ZTVN10__cxxabiv117__class_type_infoE.ptrauth to i8*), i8* inttoptr (i64 add (i64 ptrtoint ([3 x i8]* @_ZTS1A to i64), i64 -9223372036854775808) to i8*) }

auto ATI = typeid(A);
