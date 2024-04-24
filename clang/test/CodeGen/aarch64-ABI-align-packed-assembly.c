// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -fsyntax-only -triple aarch64 -target-feature +neon -S -O2 -o - %s | FileCheck %s
#include <stdarg.h>
#include <arm_neon.h>

// natural alignment 16, adjusted alignment 16
// expected alignment of copy on callee stack: 16
struct non_packed_struct {
  uint16x8_t M0; // member alignment 16
};

// natural alignment 1, adjusted alignment 1
// expected alignment of copy on callee stack: 8
struct __attribute((packed)) packed_struct {
  uint16x8_t M0; // member alignment 1, because the field is packed when the struct is packed
};

// natural alignment 1, adjusted alignment 1
// expected alignment of copy on callee stack: 8
struct packed_member {
  uint16x8_t M0 __attribute((packed)); // member alignment 1
};

// natural alignment 16, adjusted alignment 16 since __attribute((aligned (n))) sets the minimum alignment
// expected alignment of copy on callee stack: 16
struct __attribute((aligned (8))) aligned_struct_8 {
  uint16x8_t M0; // member alignment 16
};

// natural alignment 16, adjusted alignment 16
// expected alignment of copy on callee stack: 16
struct aligned_member_8 {
  uint16x8_t M0 __attribute((aligned (8))); // member alignment 16 since __attribute((aligned (n))) sets the minimum alignment
};

// natural alignment 8, adjusted alignment 8
// expected alignment of copy on callee stack: 8
#pragma pack(8)
struct pragma_packed_struct_8 {
  uint16x8_t M0; // member alignment 8 because the struct is subject to packed(8)
};

// natural alignment 4, adjusted alignment 4
// expected alignment of copy on callee stack: 8
#pragma pack(4)
struct pragma_packed_struct_4 {
  uint16x8_t M0; // member alignment 4 because the struct is subject to packed(4)
};

double gd;
void init(int, ...);

struct non_packed_struct gs_non_packed_struct;

__attribute__((noinline)) void named_arg_non_packed_struct(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct non_packed_struct s_non_packed_struct) {
// CHECK: ldr q1, [sp, #16]
    gd = d8;
    gs_non_packed_struct = s_non_packed_struct;
}

void variadic_non_packed_struct(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct non_packed_struct on_callee_stack;
  on_callee_stack = va_arg(vl, struct non_packed_struct);
}

void test_non_packed_struct() {
    struct non_packed_struct s_non_packed_struct;
    init(1, &s_non_packed_struct);

// CHECK: mov x8, #4611686018427387904        // =0x4000000000000000
// CHECK: str x8, [sp]
// CHECK: str q0, [sp, #16]
    named_arg_non_packed_struct(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_non_packed_struct);
// CHECK: str q0, [sp, #16]
    variadic_non_packed_struct(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_non_packed_struct);
}

struct packed_struct gs_packed_struct;

__attribute__((noinline)) void named_arg_packed_struct(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct packed_struct s_packed_struct) {
// CHECK: ldur q1, [sp, #8]
    gd = d8;
    gs_packed_struct = s_packed_struct;
}

void variadic_packed_struct(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct packed_struct on_callee_stack;
  on_callee_stack = va_arg(vl, struct packed_struct);
}

void test_packed_struct() {
    struct packed_struct s_packed_struct;
    init(1, &s_packed_struct);

// CHECK: mov x8, #4611686018427387904        // =0x4000000000000000
// CHECK: str x8, [sp]
// CHECK: stur q0, [sp, #8]
    named_arg_packed_struct(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_packed_struct);
// CHECK: stur q0, [sp, #8]
    variadic_packed_struct(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_packed_struct);
}

struct packed_member gs_packed_member;

__attribute__((noinline)) void named_arg_packed_member(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct packed_member s_packed_member) {
// CHECK: ldur q1, [sp, #8]
    gd = d8;
    gs_packed_member = s_packed_member;
}

void variadic_packed_member(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct packed_member on_callee_stack;
  on_callee_stack = va_arg(vl, struct packed_member);
}

void test_packed_member() {
    struct packed_member s_packed_member;
    init(1, &s_packed_member);

// CHECK: mov x8, #4611686018427387904        // =0x4000000000000000
// CHECK: str x8, [sp]
// CHECK: stur q0, [sp, #8]
    named_arg_packed_member(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_packed_member);
// CHECK: stur q0, [sp, #8]
    variadic_packed_member(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_packed_member);
}

struct aligned_struct_8 gs_aligned_struct_8;

__attribute__((noinline)) void named_arg_aligned_struct_8(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct aligned_struct_8 s_aligned_struct_8) {
// CHECK: ldr q1, [sp, #16]
    gd = d8;
    gs_aligned_struct_8 = s_aligned_struct_8;
}

void variadic_aligned_struct_8(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct aligned_struct_8 on_callee_stack;
  on_callee_stack = va_arg(vl, struct aligned_struct_8);
}

void test_aligned_struct_8() {
    struct aligned_struct_8 s_aligned_struct_8;
    init(1, &s_aligned_struct_8);

// CHECK: mov x8, #4611686018427387904        // =0x4000000000000000
// CHECK: str x8, [sp]
// CHECK: str q0, [sp, #16]
    named_arg_aligned_struct_8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_aligned_struct_8);
// CHECK: str q0, [sp, #16]
    variadic_aligned_struct_8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_aligned_struct_8);
}

struct aligned_member_8 gs_aligned_member_8;

__attribute__((noinline)) void named_arg_aligned_member_8(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct aligned_member_8 s_aligned_member_8) {
// CHECK: ldr q1, [sp, #16]
    gd = d8;
    gs_aligned_member_8 = s_aligned_member_8;
}

void variadic_aligned_member_8(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct aligned_member_8 on_callee_stack;
  on_callee_stack = va_arg(vl, struct aligned_member_8);
}

void test_aligned_member_8() {
    struct aligned_member_8 s_aligned_member_8;
    init(1, &s_aligned_member_8);

// CHECK: mov x8, #4611686018427387904        // =0x4000000000000000
// CHECK: str x8, [sp]
// CHECK: str q0, [sp, #16]
    named_arg_aligned_member_8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_aligned_member_8);
// CHECK: str q0, [sp, #16]
    variadic_aligned_member_8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_aligned_member_8);
}

struct pragma_packed_struct_8 gs_pragma_packed_struct_8;

__attribute__((noinline)) void named_arg_pragma_packed_struct_8(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct pragma_packed_struct_8 s_pragma_packed_struct_8) {
// CHECK: ldur q1, [sp, #8]
    gd = d8;
    gs_pragma_packed_struct_8 = s_pragma_packed_struct_8;
}

void variadic_pragma_packed_struct_8(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct pragma_packed_struct_8 on_callee_stack;
  on_callee_stack = va_arg(vl, struct pragma_packed_struct_8);
}

void test_pragma_packed_struct_8() {
    struct pragma_packed_struct_8 s_pragma_packed_struct_8;
    init(1, &s_pragma_packed_struct_8);

// CHECK: mov x8, #4611686018427387904        // =0x4000000000000000
// CHECK: str x8, [sp]
// CHECK: stur q0, [sp, #8]
    named_arg_pragma_packed_struct_8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_pragma_packed_struct_8);
// CHECK: stur q0, [sp, #8]
    variadic_pragma_packed_struct_8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_pragma_packed_struct_8);
}

struct pragma_packed_struct_4 gs_pragma_packed_struct_4;

__attribute__((noinline)) void named_arg_pragma_packed_struct_4(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct pragma_packed_struct_4 s_pragma_packed_struct_4) {
// CHECK: ldur q1, [sp, #8]
    gd = d8;
    gs_pragma_packed_struct_4 = s_pragma_packed_struct_4;
}

void variadic_pragma_packed_struct_4(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct pragma_packed_struct_4 on_callee_stack;
  on_callee_stack = va_arg(vl, struct pragma_packed_struct_4);
}

void test_pragma_packed_struct_4() {
    struct pragma_packed_struct_4 s_pragma_packed_struct_4;
    init(1, &s_pragma_packed_struct_4);

// CHECK: mov x8, #4611686018427387904        // =0x4000000000000000
// CHECK: str x8, [sp]
// CHECK: stur q0, [sp, #8]
    named_arg_pragma_packed_struct_4(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_pragma_packed_struct_4);
// CHECK: stur q0, [sp, #8]
    variadic_pragma_packed_struct_4(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_pragma_packed_struct_4);
}
