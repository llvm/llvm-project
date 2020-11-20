// RUN: %clang_cc1 -triple riscv64 -target-feature +d -target-abi lp64d -emit-llvm %s -o - \
// RUN:     | FileCheck %s

#include <stdint.h>

// Verify that the tracking of used GPRs and FPRs works correctly by checking
// that small integers are sign/zero extended when passed in registers.

// Doubles are passed in FPRs, so argument 'i' will be passed zero-extended
// because it will be passed in a GPR.

// CHECK: define void @f_fpr_tracking(double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, i8 zeroext %i)
void f_fpr_tracking(double a, double b, double c, double d, double e, double f,
                    double g, double h, uint8_t i) {}

// Check that fp, fp+fp, and int+fp structs are lowered correctly. These will
// be passed in FPR, FPR+FPR, or GPR+FPR regs if sufficient registers are
// available the widths are <= XLEN and FLEN, and should be expanded to
// separate arguments in IR. They are passed by the same rules for returns,
// but will be lowered to simple two-element structs if necessary (as LLVM IR
// functions cannot return multiple values).

// A struct containing just one floating-point real is passed as though it
// were a standalone floating-point real.

struct double_s { double f; };

// CHECK: define void @f_double_s_arg(double %0)
void f_double_s_arg(struct double_s a) {}

// CHECK: define double @f_ret_double_s()
struct double_s f_ret_double_s() {
  return (struct double_s){1.0};
}

// A struct containing a double and any number of zero-width bitfields is
// passed as though it were a standalone floating-point real.

struct zbf_double_s { int : 0; double f; };
struct zbf_double_zbf_s { int : 0; double f; int : 0; };

// CHECK: define void @f_zbf_double_s_arg(double %0)
void f_zbf_double_s_arg(struct zbf_double_s a) {}

// CHECK: define double @f_ret_zbf_double_s()
struct zbf_double_s f_ret_zbf_double_s() {
  return (struct zbf_double_s){1.0};
}

// CHECK: define void @f_zbf_double_zbf_s_arg(double %0)
void f_zbf_double_zbf_s_arg(struct zbf_double_zbf_s a) {}

// CHECK: define double @f_ret_zbf_double_zbf_s()
struct zbf_double_zbf_s f_ret_zbf_double_zbf_s() {
  return (struct zbf_double_zbf_s){1.0};
}

// Check that structs containing two floating point values (FLEN <= width) are
// expanded provided sufficient FPRs are available.

struct double_double_s { double f; double g; };
struct double_float_s { double f; float g; };

// CHECK: define void @f_double_double_s_arg(double %0, double %1)
void f_double_double_s_arg(struct double_double_s a) {}

// CHECK: define { double, double } @f_ret_double_double_s()
struct double_double_s f_ret_double_double_s() {
  return (struct double_double_s){1.0, 2.0};
}

// CHECK: define void @f_double_float_s_arg(double %0, float %1)
void f_double_float_s_arg(struct double_float_s a) {}

// CHECK: define { double, float } @f_ret_double_float_s()
struct double_float_s f_ret_double_float_s() {
  return (struct double_float_s){1.0, 2.0};
}

// CHECK: define void @f_double_double_s_arg_insufficient_fprs(float %a, double %b, double %c, double %d, double %e, double %f, double %g, [2 x i64] %h.coerce)
void f_double_double_s_arg_insufficient_fprs(float a, double b, double c, double d,
    double e, double f, double g, struct double_double_s h) {}

// Check that structs containing int+double values are expanded, provided
// sufficient FPRs and GPRs are available. The integer components are neither
// sign or zero-extended.

struct double_int8_s { double f; int8_t i; };
struct double_uint8_s { double f; uint8_t i; };
struct double_int32_s { double f; int32_t i; };
struct double_int64_s { double f; int64_t i; };
struct double_int128bf_s { double f; __int128_t i : 64; };
struct double_int8_zbf_s { double f; int8_t i; int : 0; };

// CHECK: define void @f_double_int8_s_arg(double %0, i8 %1)
void f_double_int8_s_arg(struct double_int8_s a) {}

// CHECK: define { double, i8 } @f_ret_double_int8_s()
struct double_int8_s f_ret_double_int8_s() {
  return (struct double_int8_s){1.0, 2};
}

// CHECK: define void @f_double_uint8_s_arg(double %0, i8 %1)
void f_double_uint8_s_arg(struct double_uint8_s a) {}

// CHECK: define { double, i8 } @f_ret_double_uint8_s()
struct double_uint8_s f_ret_double_uint8_s() {
  return (struct double_uint8_s){1.0, 2};
}

// CHECK: define void @f_double_int32_s_arg(double %0, i32 %1)
void f_double_int32_s_arg(struct double_int32_s a) {}

// CHECK: define { double, i32 } @f_ret_double_int32_s()
struct double_int32_s f_ret_double_int32_s() {
  return (struct double_int32_s){1.0, 2};
}

// CHECK: define void @f_double_int64_s_arg(double %0, i64 %1)
void f_double_int64_s_arg(struct double_int64_s a) {}

// CHECK: define { double, i64 } @f_ret_double_int64_s()
struct double_int64_s f_ret_double_int64_s() {
  return (struct double_int64_s){1.0, 2};
}

// CHECK: define void @f_double_int128bf_s_arg(double %0, i64 %1)
void f_double_int128bf_s_arg(struct double_int128bf_s a) {}

// CHECK: define { double, i64 } @f_ret_double_int128bf_s()
struct double_int128bf_s f_ret_double_int128bf_s() {
  return (struct double_int128bf_s){1.0, 2};
}

// The zero-width bitfield means the struct can't be passed according to the
// floating point calling convention.

// CHECK: define void @f_double_int8_zbf_s(double %0, i8 %1)
void f_double_int8_zbf_s(struct double_int8_zbf_s a) {}

// CHECK: define { double, i8 } @f_ret_double_int8_zbf_s()
struct double_int8_zbf_s f_ret_double_int8_zbf_s() {
  return (struct double_int8_zbf_s){1.0, 2};
}

// CHECK: define void @f_double_int8_s_arg_insufficient_gprs(i32 signext %a, i32 signext %b, i32 signext %c, i32 signext %d, i32 signext %e, i32 signext %f, i32 signext %g, i32 signext %h, [2 x i64] %i.coerce)
void f_double_int8_s_arg_insufficient_gprs(int a, int b, int c, int d, int e,
                                          int f, int g, int h, struct double_int8_s i) {}

// CHECK: define void @f_struct_double_int8_insufficient_fprs(float %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, [2 x i64] %i.coerce)
void f_struct_double_int8_insufficient_fprs(float a, double b, double c, double d,
                                           double e, double f, double g, double h, struct double_int8_s i) {}

// Complex floating-point values or structs containing a single complex
// floating-point value should be passed as if it were an fp+fp struct.

// CHECK: define void @f_doublecomplex(double %a.coerce0, double %a.coerce1)
void f_doublecomplex(double __complex__ a) {}

// CHECK: define { double, double } @f_ret_doublecomplex()
double __complex__ f_ret_doublecomplex() {
  return 1.0;
}

struct doublecomplex_s { double __complex__ c; };

// CHECK: define void @f_doublecomplex_s_arg(double %0, double %1)
void f_doublecomplex_s_arg(struct doublecomplex_s a) {}

// CHECK: define { double, double } @f_ret_doublecomplex_s()
struct doublecomplex_s f_ret_doublecomplex_s() {
  return (struct doublecomplex_s){1.0};
}

// Test single or two-element structs that need flattening. e.g. those
// containing nested structs, doubles in small arrays, zero-length structs etc.

struct doublearr1_s { double a[1]; };

// CHECK: define void @f_doublearr1_s_arg(double %0)
void f_doublearr1_s_arg(struct doublearr1_s a) {}

// CHECK: define double @f_ret_doublearr1_s()
struct doublearr1_s f_ret_doublearr1_s() {
  return (struct doublearr1_s){{1.0}};
}

struct doublearr2_s { double a[2]; };

// CHECK: define void @f_doublearr2_s_arg(double %0, double %1)
void f_doublearr2_s_arg(struct doublearr2_s a) {}

// CHECK: define { double, double } @f_ret_doublearr2_s()
struct doublearr2_s f_ret_doublearr2_s() {
  return (struct doublearr2_s){{1.0, 2.0}};
}

struct doublearr2_tricky1_s { struct { double f[1]; } g[2]; };

// CHECK: define void @f_doublearr2_tricky1_s_arg(double %0, double %1)
void f_doublearr2_tricky1_s_arg(struct doublearr2_tricky1_s a) {}

// CHECK: define { double, double } @f_ret_doublearr2_tricky1_s()
struct doublearr2_tricky1_s f_ret_doublearr2_tricky1_s() {
  return (struct doublearr2_tricky1_s){{{{1.0}}, {{2.0}}}};
}

struct doublearr2_tricky2_s { struct {}; struct { double f[1]; } g[2]; };

// CHECK: define void @f_doublearr2_tricky2_s_arg(double %0, double %1)
void f_doublearr2_tricky2_s_arg(struct doublearr2_tricky2_s a) {}

// CHECK: define { double, double } @f_ret_doublearr2_tricky2_s()
struct doublearr2_tricky2_s f_ret_doublearr2_tricky2_s() {
  return (struct doublearr2_tricky2_s){{}, {{{1.0}}, {{2.0}}}};
}

struct doublearr2_tricky3_s { union {}; struct { double f[1]; } g[2]; };

// CHECK: define void @f_doublearr2_tricky3_s_arg(double %0, double %1)
void f_doublearr2_tricky3_s_arg(struct doublearr2_tricky3_s a) {}

// CHECK: define { double, double } @f_ret_doublearr2_tricky3_s()
struct doublearr2_tricky3_s f_ret_doublearr2_tricky3_s() {
  return (struct doublearr2_tricky3_s){{}, {{{1.0}}, {{2.0}}}};
}

struct doublearr2_tricky4_s { union {}; struct { struct {}; double f[1]; } g[2]; };

// CHECK: define void @f_doublearr2_tricky4_s_arg(double %0, double %1)
void f_doublearr2_tricky4_s_arg(struct doublearr2_tricky4_s a) {}

// CHECK: define { double, double } @f_ret_doublearr2_tricky4_s()
struct doublearr2_tricky4_s f_ret_doublearr2_tricky4_s() {
  return (struct doublearr2_tricky4_s){{}, {{{}, {1.0}}, {{}, {2.0}}}};
}

// Test structs that should be passed according to the normal integer calling
// convention.

struct int_double_int_s { int a; double b; int c; };

// CHECK: define void @f_int_double_int_s_arg(%struct.int_double_int_s* %a)
void f_int_double_int_s_arg(struct int_double_int_s a) {}

// CHECK: define void @f_ret_int_double_int_s(%struct.int_double_int_s* noalias sret(%struct.int_double_int_s) align 8 %agg.result)
struct int_double_int_s f_ret_int_double_int_s() {
  return (struct int_double_int_s){1, 2.0, 3};
}

struct char_char_double_s { char a; char b; double c; };

// CHECK-LABEL: define void @f_char_char_double_s_arg([2 x i64] %a.coerce)
void f_char_char_double_s_arg(struct char_char_double_s a) {}

// CHECK: define [2 x i64] @f_ret_char_char_double_s()
struct char_char_double_s f_ret_char_char_double_s() {
  return (struct char_char_double_s){1, 2, 3.0};
}

// Unions are always passed according to the integer calling convention, even
// if they can only contain a double.

union double_u { double a; };

// CHECK: define void @f_double_u_arg(i64 %a.coerce)
void f_double_u_arg(union double_u a) {}

// CHECK: define i64 @f_ret_double_u()
union double_u f_ret_double_u() {
  return (union double_u){1.0};
}
