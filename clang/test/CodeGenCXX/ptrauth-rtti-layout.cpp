// RUN: %clang_cc1 %s -I%S -triple=arm64-apple-ios   -fptrauth-calls -std=c++11 -emit-llvm -o - | FileCheck --check-prefix=DARWIN %s
// RUN: %clang_cc1 %s -I%S -triple=aarch64-linux-gnu -fptrauth-calls -std=c++11 -emit-llvm -o - | FileCheck --check-prefix=ELF %s

#include <typeinfo>

struct A { int a; };

// DARWIN: @_ZTI1A = linkonce_odr hidden constant { ptr, ptr } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), i32 2), ptr inttoptr (i64 add (i64 ptrtoint (ptr @_ZTS1A to i64), i64 -9223372036854775808) to ptr) }
// DARWIN: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// DARWIN: @_ZTS1A = linkonce_odr hidden constant [3 x i8] c"1A\00"

// ELF: @_ZTI1A = linkonce_odr constant { ptr, ptr } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), i32 2), ptr @_ZTS1A }
// ELF: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// ELF: @_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00"

auto ATI = typeid(A);
