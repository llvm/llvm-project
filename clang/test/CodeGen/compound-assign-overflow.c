// Verify proper type emitted for compound assignments
// RUN: %clang_cc1 -ffreestanding -triple x86_64-apple-darwin10 -emit-llvm -o - %s  -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-recover=signed-integer-overflow,unsigned-integer-overflow | FileCheck %s

#include <stdint.h>

// CHECK: @[[INT:.*]] = private unnamed_addr constant { i16, i16, [22 x i8] } { i16 0, i16 11, [22 x i8] c"'int32_t' (aka 'int')\00" }
// CHECK: @[[LINE_100:.*]] = private unnamed_addr global {{.*}}, i32 100, i32 5 {{.*}} @[[INT]]
// CHECK: @[[UINT:.*]] = private unnamed_addr constant { i16, i16, [32 x i8] } { i16 0, i16 10, [32 x i8] c"'uint32_t' (aka 'unsigned int')\00" }
// CHECK: @[[LINE_200:.*]] = private unnamed_addr global {{.*}}, i32 200, i32 5 {{.*}} @[[UINT]]
// CHECK: @[[LINE_300:.*]] = private unnamed_addr global {{.*}}, i32 300, i32 5 {{.*}} @[[INT]]

int32_t x;

// CHECK: @compaddsigned
void compaddsigned(void) {
#line 100
  x += ((int32_t)1);
  // CHECK: @__ubsan_handle_add_overflow(ptr @[[LINE_100]], {{.*}})
}

// CHECK: @compaddunsigned
void compaddunsigned(void) {
#line 200
  x += ((uint32_t)1U);
  // CHECK: @__ubsan_handle_add_overflow(ptr @[[LINE_200]], {{.*}})
}

// CHECK: @compdiv
void compdiv(void) {
#line 300
  x /= x;
  // CHECK: @__ubsan_handle_divrem_overflow(ptr @[[LINE_300]], {{.*}})
}
