// RUN: %clang_cc1 -no-opaque-pointers -triple riscv32 -target-feature +d -target-feature +f -target-abi ilp32d -emit-llvm %s -o - \
// RUN:     | FileCheck %s

#include <stdint.h>

// Verify that the tracking of used GPRs and FPRs works correctly by checking
// that small integers are sign/zero extended when passed in registers.

// Doubles are passed in FPRs, so argument 'i' will be passed zero-extended
// because it will be passed in a GPR.

// CHECK: define{{.*}} void @f_fpr_tracking(double noundef %a, double noundef %b, double noundef %c, double noundef %d, double noundef %e, double noundef %f, double noundef %g, double noundef %h, i8 noundef zeroext %i)
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

// CHECK: define{{.*}} void @f_double_s_arg(double %0)
void f_double_s_arg(struct double_s a) {}

// CHECK: define{{.*}} double @f_ret_double_s()
struct double_s f_ret_double_s(void) {
  return (struct double_s){1.0};
}

// A struct containing a double and any number of zero-width bitfields is
// passed as though it were a standalone floating-point real.

struct zbf_double_s { int : 0; double f; };
struct zbf_double_zbf_s { int : 0; double f; int : 0; };

// CHECK: define{{.*}} void @f_zbf_double_s_arg(double %0)
void f_zbf_double_s_arg(struct zbf_double_s a) {}

// CHECK: define{{.*}} double @f_ret_zbf_double_s()
struct zbf_double_s f_ret_zbf_double_s(void) {
  return (struct zbf_double_s){1.0};
}

// CHECK: define{{.*}} void @f_zbf_double_zbf_s_arg(double %0)
void f_zbf_double_zbf_s_arg(struct zbf_double_zbf_s a) {}

// CHECK: define{{.*}} double @f_ret_zbf_double_zbf_s()
struct zbf_double_zbf_s f_ret_zbf_double_zbf_s(void) {
  return (struct zbf_double_zbf_s){1.0};
}

// Check that structs containing two floating point values (FLEN <= width) are
// expanded provided sufficient FPRs are available.

struct double_double_s { double f; double g; };
struct double_float_s { double f; float g; };

// CHECK: define{{.*}} void @f_double_double_s_arg(double %0, double %1)
void f_double_double_s_arg(struct double_double_s a) {}

// CHECK: define{{.*}} { double, double } @f_ret_double_double_s()
struct double_double_s f_ret_double_double_s(void) {
  return (struct double_double_s){1.0, 2.0};
}

// CHECK: define{{.*}} void @f_double_float_s_arg(double %0, float %1)
void f_double_float_s_arg(struct double_float_s a) {}

// CHECK: define{{.*}} { double, float } @f_ret_double_float_s()
struct double_float_s f_ret_double_float_s(void) {
  return (struct double_float_s){1.0, 2.0};
}

// CHECK: define{{.*}} void @f_double_double_s_arg_insufficient_fprs(float noundef %a, double noundef %b, double noundef %c, double noundef %d, double noundef %e, double noundef %f, double noundef %g, %struct.double_double_s* noundef %h)
void f_double_double_s_arg_insufficient_fprs(float a, double b, double c, double d,
    double e, double f, double g, struct double_double_s h) {}

// Check that structs containing int+double values are expanded, provided
// sufficient FPRs and GPRs are available. The integer components are neither
// sign or zero-extended.

struct double_int8_s { double f; int8_t i; };
struct double_uint8_s { double f; uint8_t i; };
struct double_int32_s { double f; int32_t i; };
struct double_int64_s { double f; int64_t i; };
struct double_int64bf_s { double f; int64_t i : 32; };
struct double_int8_zbf_s { double f; int8_t i; int : 0; };

// CHECK: define{{.*}} void @f_double_int8_s_arg(double %0, i8 %1)
void f_double_int8_s_arg(struct double_int8_s a) {}

// CHECK: define{{.*}} { double, i8 } @f_ret_double_int8_s()
struct double_int8_s f_ret_double_int8_s(void) {
  return (struct double_int8_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_double_uint8_s_arg(double %0, i8 %1)
void f_double_uint8_s_arg(struct double_uint8_s a) {}

// CHECK: define{{.*}} { double, i8 } @f_ret_double_uint8_s()
struct double_uint8_s f_ret_double_uint8_s(void) {
  return (struct double_uint8_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_double_int32_s_arg(double %0, i32 %1)
void f_double_int32_s_arg(struct double_int32_s a) {}

// CHECK: define{{.*}} { double, i32 } @f_ret_double_int32_s()
struct double_int32_s f_ret_double_int32_s(void) {
  return (struct double_int32_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_double_int64_s_arg(%struct.double_int64_s* noundef %a)
void f_double_int64_s_arg(struct double_int64_s a) {}

// CHECK: define{{.*}} void @f_ret_double_int64_s(%struct.double_int64_s* noalias sret(%struct.double_int64_s) align 8 %agg.result)
struct double_int64_s f_ret_double_int64_s(void) {
  return (struct double_int64_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_double_int64bf_s_arg(double %0, i32 %1)
void f_double_int64bf_s_arg(struct double_int64bf_s a) {}

// CHECK: define{{.*}} { double, i32 } @f_ret_double_int64bf_s()
struct double_int64bf_s f_ret_double_int64bf_s(void) {
  return (struct double_int64bf_s){1.0, 2};
}

// The zero-width bitfield means the struct can't be passed according to the
// floating point calling convention.

// CHECK: define{{.*}} void @f_double_int8_zbf_s(double %0, i8 %1)
void f_double_int8_zbf_s(struct double_int8_zbf_s a) {}

// CHECK: define{{.*}} { double, i8 } @f_ret_double_int8_zbf_s()
struct double_int8_zbf_s f_ret_double_int8_zbf_s(void) {
  return (struct double_int8_zbf_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_double_int8_s_arg_insufficient_gprs(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, i32 noundef %e, i32 noundef %f, i32 noundef %g, i32 noundef %h, %struct.double_int8_s* noundef %i)
void f_double_int8_s_arg_insufficient_gprs(int a, int b, int c, int d, int e,
                                          int f, int g, int h, struct double_int8_s i) {}

// CHECK: define{{.*}} void @f_struct_double_int8_insufficient_fprs(float noundef %a, double noundef %b, double noundef %c, double noundef %d, double noundef %e, double noundef %f, double noundef %g, double noundef %h, %struct.double_int8_s* noundef %i)
void f_struct_double_int8_insufficient_fprs(float a, double b, double c, double d,
                                           double e, double f, double g, double h, struct double_int8_s i) {}

// Complex floating-point values or structs containing a single complex
// floating-point value should be passed as if it were an fp+fp struct.

// CHECK: define{{.*}} void @f_doublecomplex(double noundef %a.coerce0, double noundef %a.coerce1)
void f_doublecomplex(double __complex__ a) {}

// CHECK: define{{.*}} { double, double } @f_ret_doublecomplex()
double __complex__ f_ret_doublecomplex(void) {
  return 1.0;
}

struct doublecomplex_s { double __complex__ c; };

// CHECK: define{{.*}} void @f_doublecomplex_s_arg(double %0, double %1)
void f_doublecomplex_s_arg(struct doublecomplex_s a) {}

// CHECK: define{{.*}} { double, double } @f_ret_doublecomplex_s()
struct doublecomplex_s f_ret_doublecomplex_s(void) {
  return (struct doublecomplex_s){1.0};
}

// Test single or two-element structs that need flattening. e.g. those
// containing nested structs, doubles in small arrays, zero-length structs etc.

struct doublearr1_s { double a[1]; };

// CHECK: define{{.*}} void @f_doublearr1_s_arg(double %0)
void f_doublearr1_s_arg(struct doublearr1_s a) {}

// CHECK: define{{.*}} double @f_ret_doublearr1_s()
struct doublearr1_s f_ret_doublearr1_s(void) {
  return (struct doublearr1_s){{1.0}};
}

struct doublearr2_s { double a[2]; };

// CHECK: define{{.*}} void @f_doublearr2_s_arg(double %0, double %1)
void f_doublearr2_s_arg(struct doublearr2_s a) {}

// CHECK: define{{.*}} { double, double } @f_ret_doublearr2_s()
struct doublearr2_s f_ret_doublearr2_s(void) {
  return (struct doublearr2_s){{1.0, 2.0}};
}

struct doublearr2_tricky1_s { struct { double f[1]; } g[2]; };

// CHECK: define{{.*}} void @f_doublearr2_tricky1_s_arg(double %0, double %1)
void f_doublearr2_tricky1_s_arg(struct doublearr2_tricky1_s a) {}

// CHECK: define{{.*}} { double, double } @f_ret_doublearr2_tricky1_s()
struct doublearr2_tricky1_s f_ret_doublearr2_tricky1_s(void) {
  return (struct doublearr2_tricky1_s){{{{1.0}}, {{2.0}}}};
}

struct doublearr2_tricky2_s { struct {}; struct { double f[1]; } g[2]; };

// CHECK: define{{.*}} void @f_doublearr2_tricky2_s_arg(double %0, double %1)
void f_doublearr2_tricky2_s_arg(struct doublearr2_tricky2_s a) {}

// CHECK: define{{.*}} { double, double } @f_ret_doublearr2_tricky2_s()
struct doublearr2_tricky2_s f_ret_doublearr2_tricky2_s(void) {
  return (struct doublearr2_tricky2_s){{}, {{{1.0}}, {{2.0}}}};
}

struct doublearr2_tricky3_s { union {}; struct { double f[1]; } g[2]; };

// CHECK: define{{.*}} void @f_doublearr2_tricky3_s_arg(double %0, double %1)
void f_doublearr2_tricky3_s_arg(struct doublearr2_tricky3_s a) {}

// CHECK: define{{.*}} { double, double } @f_ret_doublearr2_tricky3_s()
struct doublearr2_tricky3_s f_ret_doublearr2_tricky3_s(void) {
  return (struct doublearr2_tricky3_s){{}, {{{1.0}}, {{2.0}}}};
}

struct doublearr2_tricky4_s { union {}; struct { struct {}; double f[1]; } g[2]; };

// CHECK: define{{.*}} void @f_doublearr2_tricky4_s_arg(double %0, double %1)
void f_doublearr2_tricky4_s_arg(struct doublearr2_tricky4_s a) {}

// CHECK: define{{.*}} { double, double } @f_ret_doublearr2_tricky4_s()
struct doublearr2_tricky4_s f_ret_doublearr2_tricky4_s(void) {
  return (struct doublearr2_tricky4_s){{}, {{{}, {1.0}}, {{}, {2.0}}}};
}

// Test structs that should be passed according to the normal integer calling
// convention.

struct int_double_int_s { int a; double b; int c; };

// CHECK: define{{.*}} void @f_int_double_int_s_arg(%struct.int_double_int_s* noundef %a)
void f_int_double_int_s_arg(struct int_double_int_s a) {}

// CHECK: define{{.*}} void @f_ret_int_double_int_s(%struct.int_double_int_s* noalias sret(%struct.int_double_int_s) align 8 %agg.result)
struct int_double_int_s f_ret_int_double_int_s(void) {
  return (struct int_double_int_s){1, 2.0, 3};
}

struct int64_double_s { int64_t a; double b; };

// CHECK: define{{.*}} void @f_int64_double_s_arg(%struct.int64_double_s* noundef %a)
void f_int64_double_s_arg(struct int64_double_s a) {}

// CHECK: define{{.*}} void @f_ret_int64_double_s(%struct.int64_double_s* noalias sret(%struct.int64_double_s) align 8 %agg.result)
struct int64_double_s f_ret_int64_double_s(void) {
  return (struct int64_double_s){1, 2.0};
}

struct char_char_double_s { char a; char b; double c; };

// CHECK-LABEL: define{{.*}} void @f_char_char_double_s_arg(%struct.char_char_double_s* noundef %a)
void f_char_char_double_s_arg(struct char_char_double_s a) {}

// CHECK: define{{.*}} void @f_ret_char_char_double_s(%struct.char_char_double_s* noalias sret(%struct.char_char_double_s) align 8 %agg.result)
struct char_char_double_s f_ret_char_char_double_s(void) {
  return (struct char_char_double_s){1, 2, 3.0};
}

// Unions are always passed according to the integer calling convention, even
// if they can only contain a double.

union double_u { double a; };

// CHECK: define{{.*}} void @f_double_u_arg(i64 %a.coerce)
void f_double_u_arg(union double_u a) {}

// CHECK: define{{.*}} i64 @f_ret_double_u()
union double_u f_ret_double_u(void) {
  return (union double_u){1.0};
}

// Test that we don't incorrectly think double+int/double+double structs will
// be returned indirectly and thus have an off-by-one error for the number of
// GPRs available (this is an edge case when structs > 2*XLEN are still
// returned in registers). This includes complex doubles, which are treated as
// double+double structs by the ABI.

// CHECK: define{{.*}} { double, i32 } @f_ret_double_int32_s_double_int32_s_just_sufficient_gprs(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, i32 noundef %e, i32 noundef %f, i32 noundef %g, double %0, i32 %1)
struct double_int32_s f_ret_double_int32_s_double_int32_s_just_sufficient_gprs(
    int a, int b, int c, int d, int e, int f, int g, struct double_int32_s h) {
  return (struct double_int32_s){1.0, 2};
}

// CHECK: define{{.*}} { double, double } @f_ret_double_double_s_double_int32_s_just_sufficient_gprs(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, i32 noundef %e, i32 noundef %f, i32 noundef %g, double %0, i32 %1)
struct double_double_s f_ret_double_double_s_double_int32_s_just_sufficient_gprs(
    int a, int b, int c, int d, int e, int f, int g, struct double_int32_s h) {
  return (struct double_double_s){1.0, 2.0};
}

// CHECK: define{{.*}} { double, double } @f_ret_doublecomplex_double_int32_s_just_sufficient_gprs(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, i32 noundef %e, i32 noundef %f, i32 noundef %g, double %0, i32 %1)
double __complex__ f_ret_doublecomplex_double_int32_s_just_sufficient_gprs(
    int a, int b, int c, int d, int e, int f, int g, struct double_int32_s h) {
  return 1.0;
}
