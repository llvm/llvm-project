// RUN: %clang_cc1 -triple csky -target-feature +fpuv2_sf -target-feature +fpuv2_df -target-feature +hard-float-abi -target-feature +hard-float -emit-llvm %s -o - | FileCheck %s

#include <stdint.h>

// Verify that the tracking of used GPRs and FPRs works correctly by checking
// that small integers are sign/zero extended when passed in registers.

// Doubles are passed in FPRs, so argument 'i' will be passed zero-extended
// because it will be passed in a GPR.

// CHECK: define{{.*}} void @f_fpr_tracking(double noundef %a, double noundef %b, double noundef %c, double noundef %d, i8 noundef zeroext %i)
void f_fpr_tracking(double a, double b, double c, double d, uint8_t i) {}

// A struct containing just one floating-point real is passed as though it
// were a standalone floating-point real.
struct double_s {
  double f;
};

// CHECK: define{{.*}} void @f_double_s_arg(double %a.coerce)
void f_double_s_arg(struct double_s a) {}

// CHECK: define{{.*}} double @f_ret_double_s()
struct double_s f_ret_double_s(void) {
  return (struct double_s){1.0};
}

// A struct containing a double and any number of zero-width bitfields is
// passed as though it were a standalone floating-point real.

struct zbf_double_s {
  int : 0;
  double f;
};
struct zbf_double_zbf_s {
  int : 0;
  double f;
  int : 0;
};

// CHECK: define{{.*}} void @f_zbf_double_s_arg(double %a.coerce)
void f_zbf_double_s_arg(struct zbf_double_s a) {}

// CHECK: define{{.*}} double @f_ret_zbf_double_s()
struct zbf_double_s f_ret_zbf_double_s(void) {
  return (struct zbf_double_s){1.0};
}

// CHECK: define{{.*}} void @f_zbf_double_zbf_s_arg(double %a.coerce)
void f_zbf_double_zbf_s_arg(struct zbf_double_zbf_s a) {}

// CHECK: define{{.*}} double @f_ret_zbf_double_zbf_s()
struct zbf_double_zbf_s f_ret_zbf_double_zbf_s(void) {
  return (struct zbf_double_zbf_s){1.0};
}

// For argument type, the first 4*XLen parts of aggregate will be passed
// in registers, and the rest will be passed in stack.
// So we can coerce to integers directly and let backend handle it correctly.
// For return type, aggregate which <= 2*XLen will be returned in registers.
// Otherwise, aggregate will be returned indirectly.

struct double_double_s {
  double f;
  double g;
};
struct double_float_s {
  double f;
  float g;
};

// CHECK: define{{.*}} void @f_double_double_s_arg([4 x i32] %a.coerce)
void f_double_double_s_arg(struct double_double_s a) {}

// CHECK: define{{.*}} void @f_ret_double_double_s(ptr noalias sret(%struct.double_double_s) align 4 %agg.result)
struct double_double_s f_ret_double_double_s(void) {
  return (struct double_double_s){1.0, 2.0};
}

// CHECK: define{{.*}} void @f_double_float_s_arg([3 x i32] %a.coerce)
void f_double_float_s_arg(struct double_float_s a) {}

// CHECK: define{{.*}} void @f_ret_double_float_s(ptr noalias sret(%struct.double_float_s) align 4 %agg.result)
struct double_float_s f_ret_double_float_s(void) {
  return (struct double_float_s){1.0, 2.0};
}

// CHECK: define{{.*}} void @f_double_double_s_arg_insufficient_fprs(float noundef %a, double noundef %b, double noundef %c, double noundef %d, double noundef %e, double noundef %f, double noundef %g, double noundef %i, [4 x i32] %h.coerce)
void f_double_double_s_arg_insufficient_fprs(float a, double b, double c, double d,
                                             double e, double f, double g, double i, struct double_double_s h) {}

struct double_int8_s {
  double f;
  int8_t i;
};
struct double_uint8_s {
  double f;
  uint8_t i;
};
struct double_int32_s {
  double f;
  int32_t i;
};
struct double_int64_s {
  double f;
  int64_t i;
};
struct double_int64bf_s {
  double f;
  int64_t i : 32;
};
struct double_int8_zbf_s {
  double f;
  int8_t i;
  int : 0;
};

// CHECK: define{{.*}}  @f_double_int8_s_arg([3 x i32] %a.coerce)
void f_double_int8_s_arg(struct double_int8_s a) {}

// CHECK: define{{.*}} void @f_ret_double_int8_s(ptr noalias sret(%struct.double_int8_s) align 4 %agg.result)
struct double_int8_s f_ret_double_int8_s(void) {
  return (struct double_int8_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_double_uint8_s_arg([3 x i32] %a.coerce)
void f_double_uint8_s_arg(struct double_uint8_s a) {}

// CHECK: define{{.*}} void @f_ret_double_uint8_s(ptr noalias sret(%struct.double_uint8_s) align 4 %agg.result)
struct double_uint8_s f_ret_double_uint8_s(void) {
  return (struct double_uint8_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_double_int32_s_arg([3 x i32] %a.coerce)
void f_double_int32_s_arg(struct double_int32_s a) {}

// CHECK: define{{.*}} void @f_ret_double_int32_s(ptr noalias sret(%struct.double_int32_s) align 4 %agg.result)
struct double_int32_s f_ret_double_int32_s(void) {
  return (struct double_int32_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_double_int64_s_arg([4 x i32] %a.coerce)
void f_double_int64_s_arg(struct double_int64_s a) {}

// CHECK: define{{.*}} void @f_ret_double_int64_s(ptr noalias sret(%struct.double_int64_s) align 4 %agg.result)
struct double_int64_s f_ret_double_int64_s(void) {
  return (struct double_int64_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_double_int64bf_s_arg([3 x i32] %a.coerce)
void f_double_int64bf_s_arg(struct double_int64bf_s a) {}

// CHECK: define{{.*}} void @f_ret_double_int64bf_s(ptr noalias sret(%struct.double_int64bf_s) align 4 %agg.result)
struct double_int64bf_s f_ret_double_int64bf_s(void) {
  return (struct double_int64bf_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_double_int8_zbf_s([3 x i32] %a.coerce)
void f_double_int8_zbf_s(struct double_int8_zbf_s a) {}

// CHECK: define{{.*}} void @f_ret_double_int8_zbf_s(ptr noalias sret(%struct.double_int8_zbf_s) align 4 %agg.result)
struct double_int8_zbf_s f_ret_double_int8_zbf_s(void) {
  return (struct double_int8_zbf_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_double_int8_s_arg_insufficient_gprs(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, i32 noundef %e, i32 noundef %f, i32 noundef %g, i32 noundef %h, [3 x i32] %i.coerce)
void f_double_int8_s_arg_insufficient_gprs(int a, int b, int c, int d, int e,
                                           int f, int g, int h, struct double_int8_s i) {}

// CHECK: define{{.*}} void @f_struct_double_int8_insufficient_fprs(float noundef %a, double noundef %b, double noundef %c, double noundef %d, double noundef %e, double noundef %f, double noundef %g, double noundef %h, [3 x i32] %i.coerce)
void f_struct_double_int8_insufficient_fprs(float a, double b, double c, double d,
                                            double e, double f, double g, double h, struct double_int8_s i) {}

// Complex floating-point values are special in passing argument,
// and it's not same as structs containing a single complex.
// Complex floating-point value should be passed in two consecutive fprs.
// But the return process is same as struct.

// CHECK: define{{.*}} void @f_doublecomplex(double noundef %a.coerce0, double noundef %a.coerce1)
void f_doublecomplex(double __complex__ a) {}

// CHECK: define{{.*}} void @f_ret_doublecomplex(ptr noalias sret({ double, double }) align 4 %agg.result)
double __complex__ f_ret_doublecomplex(void) {
  return 1.0;
}

struct doublecomplex_s {
  double __complex__ c;
};

// CHECK: define{{.*}} void @f_doublecomplex_s_arg([4 x i32] %a.coerce)
void f_doublecomplex_s_arg(struct doublecomplex_s a) {}

// CHECK: define{{.*}} void @f_ret_doublecomplex_s(ptr noalias sret(%struct.doublecomplex_s) align 4 %agg.result)
struct doublecomplex_s f_ret_doublecomplex_s(void) {
  return (struct doublecomplex_s){1.0};
}

// Test single or two-element structs that need flattening. e.g. those
// containing nested structs, doubles in small arrays, zero-length structs etc.

struct doublearr1_s {
  double a[1];
};

// CHECK: define{{.*}} void @f_doublearr1_s_arg(double %a.coerce)
void f_doublearr1_s_arg(struct doublearr1_s a) {}

// CHECK: define{{.*}} double @f_ret_doublearr1_s()
struct doublearr1_s f_ret_doublearr1_s(void) {
  return (struct doublearr1_s){{1.0}};
}

struct doublearr2_s {
  double a[2];
};

// CHECK: define{{.*}} void @f_doublearr2_s_arg([4 x i32] %a.coerce)
void f_doublearr2_s_arg(struct doublearr2_s a) {}

// CHECK: define{{.*}} void @f_ret_doublearr2_s(ptr noalias sret(%struct.doublearr2_s) align 4 %agg.result)
struct doublearr2_s f_ret_doublearr2_s(void) {
  return (struct doublearr2_s){{1.0, 2.0}};
}

struct doublearr2_tricky1_s {
  struct {
    double f[1];
  } g[2];
};

// CHECK: define{{.*}} void @f_doublearr2_tricky1_s_arg([4 x i32] %a.coerce)
void f_doublearr2_tricky1_s_arg(struct doublearr2_tricky1_s a) {}

// CHECK: define{{.*}} void @f_ret_doublearr2_tricky1_s(ptr noalias sret(%struct.doublearr2_tricky1_s) align 4 %agg.result)
struct doublearr2_tricky1_s f_ret_doublearr2_tricky1_s(void) {
  return (struct doublearr2_tricky1_s){{{{1.0}}, {{2.0}}}};
}

struct doublearr2_tricky2_s {
  struct {};
  struct {
    double f[1];
  } g[2];
};

// CHECK: define{{.*}} void @f_doublearr2_tricky2_s_arg([4 x i32] %a.coerce)
void f_doublearr2_tricky2_s_arg(struct doublearr2_tricky2_s a) {}

// CHECK: define{{.*}} void @f_ret_doublearr2_tricky2_s(ptr noalias sret(%struct.doublearr2_tricky2_s) align 4 %agg.result)
struct doublearr2_tricky2_s f_ret_doublearr2_tricky2_s(void) {
  return (struct doublearr2_tricky2_s){{}, {{{1.0}}, {{2.0}}}};
}

struct doublearr2_tricky3_s {
  union {};
  struct {
    double f[1];
  } g[2];
};

// CHECK: define{{.*}} void @f_doublearr2_tricky3_s_arg([4 x i32] %a.coerce)
void f_doublearr2_tricky3_s_arg(struct doublearr2_tricky3_s a) {}

// CHECK: define{{.*}} void @f_ret_doublearr2_tricky3_s(ptr noalias sret(%struct.doublearr2_tricky3_s) align 4 %agg.result)
struct doublearr2_tricky3_s f_ret_doublearr2_tricky3_s(void) {
  return (struct doublearr2_tricky3_s){{}, {{{1.0}}, {{2.0}}}};
}

struct doublearr2_tricky4_s {
  union {};
  struct {
    struct {};
    double f[1];
  } g[2];
};

// CHECK: define{{.*}} void @f_doublearr2_tricky4_s_arg([4 x i32] %a.coerce)
void f_doublearr2_tricky4_s_arg(struct doublearr2_tricky4_s a) {}

// CHECK: define{{.*}} void @f_ret_doublearr2_tricky4_s(ptr noalias sret(%struct.doublearr2_tricky4_s) align 4 %agg.result)
struct doublearr2_tricky4_s f_ret_doublearr2_tricky4_s(void) {
  return (struct doublearr2_tricky4_s){{}, {{{}, {1.0}}, {{}, {2.0}}}};
}

struct int_double_int_s {
  int a;
  double b;
  int c;
};

// CHECK: define{{.*}} void @f_int_double_int_s_arg([4 x i32] %a.coerce)
void f_int_double_int_s_arg(struct int_double_int_s a) {}

// CHECK: define{{.*}} void @f_ret_int_double_int_s(ptr noalias sret(%struct.int_double_int_s) align 4 %agg.result)
struct int_double_int_s f_ret_int_double_int_s(void) {
  return (struct int_double_int_s){1, 2.0, 3};
}

struct int64_double_s {
  int64_t a;
  double b;
};

// CHECK: define{{.*}} void @f_int64_double_s_arg([4 x i32] %a.coerce)
void f_int64_double_s_arg(struct int64_double_s a) {}

// CHECK: define{{.*}} void @f_ret_int64_double_s(ptr noalias sret(%struct.int64_double_s) align 4 %agg.result)
struct int64_double_s f_ret_int64_double_s(void) {
  return (struct int64_double_s){1, 2.0};
}

struct char_char_double_s {
  char a;
  char b;
  double c;
};

// CHECK-LABEL: define{{.*}} void @f_char_char_double_s_arg([3 x i32] %a.coerce)
void f_char_char_double_s_arg(struct char_char_double_s a) {}

// CHECK: define{{.*}} void @f_ret_char_char_double_s(ptr noalias sret(%struct.char_char_double_s) align 4 %agg.result)
struct char_char_double_s f_ret_char_char_double_s(void) {
  return (struct char_char_double_s){1, 2, 3.0};
}

// A union containing just one floating-point real can not be  passed as though it
// were a standalone floating-point real.
union double_u {
  double a;
};

// CHECK: define{{.*}} void @f_double_u_arg([2 x i32] %a.coerce)
void f_double_u_arg(union double_u a) {}

// CHECK: define{{.*}} [2 x i32] @f_ret_double_u()
union double_u f_ret_double_u(void) {
  return (union double_u){1.0};
}

// CHECK: define{{.*}} void @f_ret_double_int32_s_double_int32_s_just_sufficient_gprs(ptr noalias sret(%struct.double_int32_s) align 4 %agg.result, i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, i32 noundef %e, i32 noundef %f, i32 noundef %g, [3 x i32] %h.coerce)
struct double_int32_s f_ret_double_int32_s_double_int32_s_just_sufficient_gprs(
    int a, int b, int c, int d, int e, int f, int g, struct double_int32_s h) {
  return (struct double_int32_s){1.0, 2};
}

// CHECK: define{{.*}} void @f_ret_double_double_s_double_int32_s_just_sufficient_gprs(ptr noalias sret(%struct.double_double_s) align 4 %agg.result, i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, i32 noundef %e, i32 noundef %f, i32 noundef %g, [3 x i32] %h.coerce)
struct double_double_s f_ret_double_double_s_double_int32_s_just_sufficient_gprs(
    int a, int b, int c, int d, int e, int f, int g, struct double_int32_s h) {
  return (struct double_double_s){1.0, 2.0};
}

// CHECK: define{{.*}} void @f_ret_doublecomplex_double_int32_s_just_sufficient_gprs(ptr noalias sret({ double, double }) align 4 %agg.result, i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, i32 noundef %e, i32 noundef %f, i32 noundef %g, [3 x i32] %h.coerce)
double __complex__ f_ret_doublecomplex_double_int32_s_just_sufficient_gprs(
    int a, int b, int c, int d, int e, int f, int g, struct double_int32_s h) {
  return 1.0;
}

struct tiny {
  uint8_t a, b, c, d;
};

struct small {
  int32_t a, *b;
};

struct small_aligned {
  int64_t a;
};

struct large {
  int32_t a, b, c, d;
};

// Ensure that scalars passed on the stack are still determined correctly in
// the presence of large return values that consume a register due to the need
// to pass a pointer.

// CHECK-LABEL: define{{.*}} void @f_scalar_stack_2(ptr noalias sret(%struct.large) align 4 %agg.result, float noundef %a, i64 noundef %b, double noundef %c, double noundef %d, i8 noundef zeroext %e, i8 noundef signext %f, i8 noundef zeroext %g)
struct large f_scalar_stack_2(float a, int64_t b, double c, long double d,
                              uint8_t e, int8_t f, uint8_t g) {
  return (struct large){a, e, f, g};
}

// Aggregates and >=XLen scalars passed on the stack should be lowered just as
// they would be if passed via registers.

// CHECK-LABEL: define{{.*}} void @f_scalar_stack_3(double noundef %a, i64 noundef %b, double noundef %c, i64 noundef %d, i32 noundef %e, i64 noundef %f, float noundef %g, double noundef %h, double noundef %i)
void f_scalar_stack_3(double a, int64_t b, double c, int64_t d, int e,
                      int64_t f, float g, double h, long double i) {}

// CHECK-LABEL: define{{.*}} void @f_agg_stack(double noundef %a, i64 noundef %b, double noundef %c, i64 noundef %d, i32 %e.coerce, [2 x i32] %f.coerce, i64 %g.coerce, [4 x i32] %h.coerce)
void f_agg_stack(double a, int64_t b, double c, int64_t d, struct tiny e,
                 struct small f, struct small_aligned g, struct large h) {}
