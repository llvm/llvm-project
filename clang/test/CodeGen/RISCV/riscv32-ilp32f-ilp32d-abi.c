// RUN: %clang_cc1 -triple riscv32 -target-feature +f -target-abi ilp32f -emit-llvm %s -o - \
// RUN:     | FileCheck %s
// RUN: %clang_cc1 -triple riscv32 -target-feature +d -target-feature +f -target-abi ilp32d -emit-llvm %s -o - \
// RUN:     | FileCheck %s

#include <stdint.h>

// Verify that the tracking of used GPRs and FPRs works correctly by checking
// that small integers are sign/zero extended when passed in registers.

// Floats are passed in FPRs, so argument 'i' will be passed zero-extended
// because it will be passed in a GPR.

// CHECK: define{{.*}} void @f_fpr_tracking(float noundef %a, float noundef %b, float noundef %c, float noundef %d, float noundef %e, float noundef %f, float noundef %g, float noundef %h, i8 noundef zeroext %i)
void f_fpr_tracking(float a, float b, float c, float d, float e, float f,
                    float g, float h, uint8_t i) {}

// Check that fp, fp+fp, and int+fp structs are lowered correctly. These will
// be passed in FPR, FPR+FPR, or GPR+FPR regs if sufficient registers are
// available the widths are <= XLEN and FLEN, and should be expanded to
// separate arguments in IR. They are passed by the same rules for returns,
// but will be lowered to simple two-element structs if necessary (as LLVM IR
// functions cannot return multiple values).

// A struct containing just one floating-point real is passed as though it
// were a standalone floating-point real.

struct float_s { float f; };

// CHECK: define{{.*}} void @f_float_s_arg(float %0)
void f_float_s_arg(struct float_s a) {}

// CHECK: define{{.*}} float @f_ret_float_s()
struct float_s f_ret_float_s() {
  return (struct float_s){1.0};
}

// A struct containing a float and any number of zero-width bitfields is
// passed as though it were a standalone floating-point real.

struct zbf_float_s { int : 0; float f; };
struct zbf_float_zbf_s { int : 0; float f; int : 0; };

// CHECK: define{{.*}} void @f_zbf_float_s_arg(float %0)
void f_zbf_float_s_arg(struct zbf_float_s a) {}

// CHECK: define{{.*}} float @f_ret_zbf_float_s()
struct zbf_float_s f_ret_zbf_float_s() {
  return (struct zbf_float_s){1.0};
}

// CHECK: define{{.*}} void @f_zbf_float_zbf_s_arg(float %0)
void f_zbf_float_zbf_s_arg(struct zbf_float_zbf_s a) {}

// CHECK: define{{.*}} float @f_ret_zbf_float_zbf_s()
struct zbf_float_zbf_s f_ret_zbf_float_zbf_s() {
  return (struct zbf_float_zbf_s){1.0};
}

// Check that structs containing two float values (FLEN <= width) are expanded
// provided sufficient FPRs are available.

struct float_float_s { float f; float g; };

// CHECK: define{{.*}} void @f_float_float_s_arg(float %0, float %1)
void f_float_float_s_arg(struct float_float_s a) {}

// CHECK: define{{.*}} { float, float } @f_ret_float_float_s()
struct float_float_s f_ret_float_float_s() {
  return (struct float_float_s){1.0, 2.0};
}

// CHECK: define{{.*}} void @f_float_float_s_arg_insufficient_fprs(float noundef %a, float noundef %b, float noundef %c, float noundef %d, float noundef %e, float noundef %f, float noundef %g, [2 x i32] %h.coerce)
void f_float_float_s_arg_insufficient_fprs(float a, float b, float c, float d,
    float e, float f, float g, struct float_float_s h) {}

// Check that structs containing int+float values are expanded, provided
// sufficient FPRs and GPRs are available. The integer components are neither
// sign or zero-extended.

struct float_int8_s { float f; int8_t i; };
struct float_uint8_s { float f; uint8_t i; };
struct float_int32_s { float f; int32_t i; };
struct float_int64_s { float f; int64_t i; };
struct float_int64bf_s { float f; int64_t i : 32; };
struct float_int8_zbf_s { float f; int8_t i; int : 0; };

// CHECK: define{{.*}} void @f_float_int8_s_arg(float %0, i8 %1)
void f_float_int8_s_arg(struct float_int8_s a) {}

// CHECK: define{{.*}} { float, i8 } @f_ret_float_int8_s()
struct float_int8_s f_ret_float_int8_s() {
  return (struct float_int8_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_float_uint8_s_arg(float %0, i8 %1)
void f_float_uint8_s_arg(struct float_uint8_s a) {}

// CHECK: define{{.*}} { float, i8 } @f_ret_float_uint8_s()
struct float_uint8_s f_ret_float_uint8_s() {
  return (struct float_uint8_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_float_int32_s_arg(float %0, i32 %1)
void f_float_int32_s_arg(struct float_int32_s a) {}

// CHECK: define{{.*}} { float, i32 } @f_ret_float_int32_s()
struct float_int32_s f_ret_float_int32_s() {
  return (struct float_int32_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_float_int64_s_arg(%struct.float_int64_s* noundef %a)
void f_float_int64_s_arg(struct float_int64_s a) {}

// CHECK: define{{.*}} void @f_ret_float_int64_s(%struct.float_int64_s* noalias sret(%struct.float_int64_s) align 8 %agg.result)
struct float_int64_s f_ret_float_int64_s() {
  return (struct float_int64_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_float_int64bf_s_arg(float %0, i32 %1)
void f_float_int64bf_s_arg(struct float_int64bf_s a) {}

// CHECK: define{{.*}} { float, i32 } @f_ret_float_int64bf_s()
struct float_int64bf_s f_ret_float_int64bf_s() {
  return (struct float_int64bf_s){1.0, 2};
}

// The zero-width bitfield means the struct can't be passed according to the
// floating point calling convention.

// CHECK: define{{.*}} void @f_float_int8_zbf_s(float %0, i8 %1)
void f_float_int8_zbf_s(struct float_int8_zbf_s a) {}

// CHECK: define{{.*}} { float, i8 } @f_ret_float_int8_zbf_s()
struct float_int8_zbf_s f_ret_float_int8_zbf_s() {
  return (struct float_int8_zbf_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_float_int8_s_arg_insufficient_gprs(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, i32 noundef %e, i32 noundef %f, i32 noundef %g, i32 noundef %h, [2 x i32] %i.coerce)
void f_float_int8_s_arg_insufficient_gprs(int a, int b, int c, int d, int e,
                                          int f, int g, int h, struct float_int8_s i) {}

// CHECK: define{{.*}} void @f_struct_float_int8_insufficient_fprs(float noundef %a, float noundef %b, float noundef %c, float noundef %d, float noundef %e, float noundef %f, float noundef %g, float noundef %h, [2 x i32] %i.coerce)
void f_struct_float_int8_insufficient_fprs(float a, float b, float c, float d,
                                           float e, float f, float g, float h, struct float_int8_s i) {}

// Complex floating-point values or structs containing a single complex
// floating-point value should be passed as if it were an fp+fp struct.

// CHECK: define{{.*}} void @f_floatcomplex(float noundef %a.coerce0, float noundef %a.coerce1)
void f_floatcomplex(float __complex__ a) {}

// CHECK: define{{.*}} { float, float } @f_ret_floatcomplex()
float __complex__ f_ret_floatcomplex() {
  return 1.0;
}

struct floatcomplex_s { float __complex__ c; };

// CHECK: define{{.*}} void @f_floatcomplex_s_arg(float %0, float %1)
void f_floatcomplex_s_arg(struct floatcomplex_s a) {}

// CHECK: define{{.*}} { float, float } @f_ret_floatcomplex_s()
struct floatcomplex_s f_ret_floatcomplex_s() {
  return (struct floatcomplex_s){1.0};
}

// Test single or two-element structs that need flattening. e.g. those
// containing nested structs, floats in small arrays, zero-length structs etc.

struct floatarr1_s { float a[1]; };

// CHECK: define{{.*}} void @f_floatarr1_s_arg(float %0)
void f_floatarr1_s_arg(struct floatarr1_s a) {}

// CHECK: define{{.*}} float @f_ret_floatarr1_s()
struct floatarr1_s f_ret_floatarr1_s() {
  return (struct floatarr1_s){{1.0}};
}

struct floatarr2_s { float a[2]; };

// CHECK: define{{.*}} void @f_floatarr2_s_arg(float %0, float %1)
void f_floatarr2_s_arg(struct floatarr2_s a) {}

// CHECK: define{{.*}} { float, float } @f_ret_floatarr2_s()
struct floatarr2_s f_ret_floatarr2_s() {
  return (struct floatarr2_s){{1.0, 2.0}};
}

struct floatarr2_tricky1_s { struct { float f[1]; } g[2]; };

// CHECK: define{{.*}} void @f_floatarr2_tricky1_s_arg(float %0, float %1)
void f_floatarr2_tricky1_s_arg(struct floatarr2_tricky1_s a) {}

// CHECK: define{{.*}} { float, float } @f_ret_floatarr2_tricky1_s()
struct floatarr2_tricky1_s f_ret_floatarr2_tricky1_s() {
  return (struct floatarr2_tricky1_s){{{{1.0}}, {{2.0}}}};
}

struct floatarr2_tricky2_s { struct {}; struct { float f[1]; } g[2]; };

// CHECK: define{{.*}} void @f_floatarr2_tricky2_s_arg(float %0, float %1)
void f_floatarr2_tricky2_s_arg(struct floatarr2_tricky2_s a) {}

// CHECK: define{{.*}} { float, float } @f_ret_floatarr2_tricky2_s()
struct floatarr2_tricky2_s f_ret_floatarr2_tricky2_s() {
  return (struct floatarr2_tricky2_s){{}, {{{1.0}}, {{2.0}}}};
}

struct floatarr2_tricky3_s { union {}; struct { float f[1]; } g[2]; };

// CHECK: define{{.*}} void @f_floatarr2_tricky3_s_arg(float %0, float %1)
void f_floatarr2_tricky3_s_arg(struct floatarr2_tricky3_s a) {}

// CHECK: define{{.*}} { float, float } @f_ret_floatarr2_tricky3_s()
struct floatarr2_tricky3_s f_ret_floatarr2_tricky3_s() {
  return (struct floatarr2_tricky3_s){{}, {{{1.0}}, {{2.0}}}};
}

struct floatarr2_tricky4_s { union {}; struct { struct {}; float f[1]; } g[2]; };

// CHECK: define{{.*}} void @f_floatarr2_tricky4_s_arg(float %0, float %1)
void f_floatarr2_tricky4_s_arg(struct floatarr2_tricky4_s a) {}

// CHECK: define{{.*}} { float, float } @f_ret_floatarr2_tricky4_s()
struct floatarr2_tricky4_s f_ret_floatarr2_tricky4_s() {
  return (struct floatarr2_tricky4_s){{}, {{{}, {1.0}}, {{}, {2.0}}}};
}

// Test structs that should be passed according to the normal integer calling
// convention.

struct int_float_int_s { int a; float b; int c; };

// CHECK: define{{.*}} void @f_int_float_int_s_arg(%struct.int_float_int_s* noundef %a)
void f_int_float_int_s_arg(struct int_float_int_s a) {}

// CHECK: define{{.*}} void @f_ret_int_float_int_s(%struct.int_float_int_s* noalias sret(%struct.int_float_int_s) align 4 %agg.result)
struct int_float_int_s f_ret_int_float_int_s() {
  return (struct int_float_int_s){1, 2.0, 3};
}

struct int64_float_s { int64_t a; float b; };

// CHECK: define{{.*}} void @f_int64_float_s_arg(%struct.int64_float_s* noundef %a)
void f_int64_float_s_arg(struct int64_float_s a) {}

// CHECK: define{{.*}} void @f_ret_int64_float_s(%struct.int64_float_s* noalias sret(%struct.int64_float_s) align 8 %agg.result)
struct int64_float_s f_ret_int64_float_s() {
  return (struct int64_float_s){1, 2.0};
}

struct char_char_float_s { char a; char b; float c; };

// CHECK-LABEL: define{{.*}} void @f_char_char_float_s_arg([2 x i32] %a.coerce)
void f_char_char_float_s_arg(struct char_char_float_s a) {}

// CHECK: define{{.*}} [2 x i32] @f_ret_char_char_float_s()
struct char_char_float_s f_ret_char_char_float_s() {
  return (struct char_char_float_s){1, 2, 3.0};
}

// Unions are always passed according to the integer calling convention, even
// if they can only contain a float.

union float_u { float a; };

// CHECK: define{{.*}} void @f_float_u_arg(i32 %a.coerce)
void f_float_u_arg(union float_u a) {}

// CHECK: define{{.*}} i32 @f_ret_float_u()
union float_u f_ret_float_u() {
  return (union float_u){1.0};
}
