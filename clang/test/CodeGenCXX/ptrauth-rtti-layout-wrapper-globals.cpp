// RUN: %clang_cc1 %s -I%S -triple=arm64-apple-ios -fptrauth-calls -std=c++11 -emit-llvm -o - | FileCheck %s
#include <typeinfo>

struct A { int a; };

// CHECK: @_ZTI1A = linkonce_odr hidden constant { ptr, ptr } { ptr @_ZTVN10__cxxabiv117__class_type_infoE.ptrauth, ptr inttoptr (i64 add (i64 ptrtoint (ptr @_ZTS1A to i64), i64 -9223372036854775808) to ptr) }
// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE.ptrauth = private constant { ptr, i32, i64, i64 } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), i32 2, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @_ZTS1A = linkonce_odr hidden constant [3 x i8] c"1A\00"

auto ATI = typeid(A);
