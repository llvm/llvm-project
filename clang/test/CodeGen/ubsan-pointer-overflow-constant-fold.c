// REQUIRES: msp430-registered-target
// RUN: %clang_cc1 -triple msp430 -emit-llvm -fsanitize=pointer-overflow -O0 %s -o - | FileCheck %s

// Verify that pointer-overflow checks are emitted correctly when the GEP
// offset is a constant that overflows the 16-bit intptr_t. Previously this
// triggered an assertion in EmitCheckedInBoundsGEP because the code assumed
// a constant-folded offset could never have an overflow.

struct sensor_reading { long timestamp; long value; };

struct sensor_reading *readings[];

// CHECK-LABEL: define {{.*}}@test_constant_offset_overflow
// CHECK: getelementptr inbounds %struct.sensor_reading
// CHECK: call void @__ubsan_handle_pointer_overflow
void test_constant_offset_overflow(void) {
  // sizeof(struct sensor_reading) == 8, index 4096: 4096 * 8 = 32768 overflows i16.
  readings[0][4096];
}

// CHECK-LABEL: define {{.*}}@test_constant_offset_overflow_wraps_to_zero
// CHECK: getelementptr inbounds %struct.sensor_reading
// CHECK: call void @__ubsan_handle_pointer_overflow
void test_constant_offset_overflow_wraps_to_zero(void) {
  // sizeof(struct sensor_reading) == 8, index 8192: 8192 * 8 = 65536 wraps to 0 in i16.
  readings[0][8192];
}

// CHECK-LABEL: define {{.*}}@test_zero_offset_no_overflow
// CHECK-NOT: __ubsan_handle_pointer_overflow
// CHECK: ret void
void test_zero_offset_no_overflow(void) {
  // Genuine zero offset should not emit a check.
  readings[0][0];
}
