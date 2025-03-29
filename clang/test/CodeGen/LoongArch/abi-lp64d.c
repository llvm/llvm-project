// RUN: %clang_cc1 -triple loongarch64 -target-feature +f -target-feature +d -target-abi lp64d \
// RUN:   -emit-llvm %s -o - | FileCheck %s

/// This test checks the calling convention of the lp64d ABI.

#include <stddef.h>
#include <stdint.h>

/// Part 0: C Data Types and Alignment.

/// `char` datatype is signed by default.
/// In most cases, the unsigned integer data types are zero-extended when stored
/// in general-purpose register, and the signed integer data types are
/// sign-extended. However, in the LP64D ABI, unsigned 32-bit types, such as
/// unsigned int, are stored in general-purpose registers as proper sign
/// extensions of their 32-bit values.

// CHECK-LABEL: define{{.*}} zeroext i1 @check_bool()
_Bool check_bool() { return 0; }

// CHECK-LABEL: define{{.*}} signext i8 @check_char()
char check_char() { return 0; }

// CHECK-LABEL: define{{.*}} signext i16 @check_short()
short check_short() { return 0; }

// CHECK-LABEL: define{{.*}} signext i32 @check_int()
int check_int() { return 0; }

// CHECK-LABEL: define{{.*}} i64 @check_long()
long check_long() { return 0; }

// CHECK-LABEL: define{{.*}} i64 @check_longlong()
long long check_longlong() { return 0; }

// CHECK-LABEL: define{{.*}} zeroext i8 @check_uchar()
unsigned char check_uchar() { return 0; }

// CHECK-LABEL: define{{.*}} zeroext i16 @check_ushort()
unsigned short check_ushort() { return 0; }

// CHECK-LABEL: define{{.*}} signext i32 @check_uint()
unsigned int check_uint() { return 0; }

// CHECK-LABEL: define{{.*}} i64 @check_ulong()
unsigned long check_ulong() { return 0; }

// CHECK-LABEL: define{{.*}} i64 @check_ulonglong()
unsigned long long check_ulonglong() { return 0; }

// CHECK-LABEL: define{{.*}} float @check_float()
float check_float() { return 0; }

// CHECK-LABEL: define{{.*}} double @check_double()
double check_double() { return 0; }

// CHECK-LABEL: define{{.*}} fp128 @check_longdouble()
long double check_longdouble() { return 0; }

/// Part 1: Scalar arguments and return value.

/// 1. 1 < WOA <= GRLEN
/// a. Argument is passed in a single argument register, or on the stack by
/// value if none is available.
/// i. If the argument is floating-point type, the argument is passed in FAR. if
/// no FAR is available, it’s passed in GAR. If no GAR is available, it’s
/// passed on the stack. When passed in registers or on the stack,
/// floating-point types narrower than GRLEN bits are widened to GRLEN bits,
/// with the upper bits undefined.
/// ii. If the argument is integer or pointer type, the argument is passed in
/// GAR. If no GAR is available, it’s passed on the stack. When passed in
/// registers or on the stack, the unsigned integer scalars narrower than GRLEN
/// bits are zero-extended to GRLEN bits, and the signed integer scalars are
/// sign-extended.
/// 2. GRLEN < WOA ≤ 2 × GRLEN
/// a. The argument is passed in a pair of GAR, with the low-order GRLEN bits in
/// the lower-numbered register and the high-order GRLEN bits in the
/// higher-numbered register. If exactly one register is available, the
/// low-order GRLEN bits are passed in the register and the high-order GRLEN
/// bits are passed on the stack. If no GAR is available, it’s passed on the
/// stack.

/// Note that most of these conventions are handled by the backend, so here we
/// only check the correctness of argument (or return value)'s sign/zero
/// extension attribute.

// CHECK-LABEL: define{{.*}} signext i32 @f_scalar(i1 noundef zeroext %a, i8 noundef signext %b, i8 noundef zeroext %c, i16 noundef signext %d, i16 noundef zeroext %e, i32 noundef signext %f, i32 noundef signext %g, i64 noundef %h, i1 noundef zeroext %i, i8 noundef signext %j, i8 noundef zeroext %k, i16 noundef signext %l, i16 noundef zeroext %m, i32 noundef signext %n, i32 noundef signext %o, i64 noundef %p)
int f_scalar(_Bool a, int8_t b, uint8_t c, int16_t d, uint16_t e, int32_t f,
             uint32_t g, int64_t h, _Bool i, int8_t j, uint8_t k, int16_t l,
             uint16_t m, int32_t n, uint32_t o, int64_t p) {
  return 0;
}

/// Part 2: Structure arguments and return value.

/// Empty structures are ignored by C compilers which support them as a
/// non-standard extension(same as union arguments and return values). Bits
/// unused due to padding, and bits past the end of a structure whose size in
/// bits is not divisible by GRLEN, are undefined. And the layout of the
/// structure on the stack is consistent with that in memory.

/// Check empty structs are ignored.

struct empty_s {};

// CHECK-LABEL: define{{.*}} void @f_empty_s()
struct empty_s f_empty_s(struct empty_s x) {
  return x;
}

/// 1. 0 < WOA ≤ GRLEN
/// a. The structure has only fixed-point members. If there is an available GAR,
/// the structure is passed through the GAR by value passing; If no GAR is
/// available, it’s passed on the stack.

struct i16x4_s {
  int16_t a, b, c, d;
};

// CHECK-LABEL: define{{.*}} i64 @f_i16x4_s(i64 %x.coerce)
struct i16x4_s f_i16x4_s(struct i16x4_s x) {
  return x;
}

/// b. The structure has only floating-point members:
/// i. One floating-point member. The argument is passed in a FAR; If no FAR is
/// available, the value is passed in a GAR; if no GAR is available, the value
/// is passed on the stack.

struct f32x1_s {
  float a;
};

struct f64x1_s {
  double a;
};

// CHECK-LABEL: define{{.*}} float @f_f32x1_s(float %0)
struct f32x1_s f_f32x1_s(struct f32x1_s x) {
  return x;
}

// CHECK-LABEL: define{{.*}} double @f_f64x1_s(double %0)
struct f64x1_s f_f64x1_s(struct f64x1_s x) {
  return x;
}

/// ii. Two floating-point members. The argument is passed in a pair of
/// available FAR, with the low-order float member bits in the lower-numbered
/// FAR and the high-order float member bits in the higher-numbered FAR. If the
/// number of available FAR is less than 2, it’s passed in a GAR, and passed on
/// the stack if no GAR is available.

struct f32x2_s {
  float a, b;
};

// CHECK-LABEL: define{{.*}} { float, float } @f_f32x2_s(float %0, float %1)
struct f32x2_s f_f32x2_s(struct f32x2_s x) {
  return x;
}

/// c. The structure has both fixed-point and floating-point members, i.e. the
/// structure has one float member and...
/// i. Multiple fixed-point members. If there are available GAR, the structure
/// is passed in a GAR, and passed on the stack if no GAR is available.

struct f32x1_i16x2_s {
  float a;
  int16_t b, c;
};

// CHECK-LABEL: define{{.*}} i64 @f_f32x1_i16x2_s(i64 %x.coerce)
struct f32x1_i16x2_s f_f32x1_i16x2_s(struct f32x1_i16x2_s x) {
  return x;
}

/// ii. Only one fixed-point member. If one FAR and one GAR are available, the
/// floating-point member of the structure is passed in the FAR, and the integer
/// member of the structure is passed in the GAR; If no floating-point register
/// but one GAR is available, it’s passed in GAR; If no GAR is available, it’s
/// passed on the stack.

struct f32x1_i32x1_s {
  float a;
  int32_t b;
};

// CHECK-LABEL: define{{.*}} { float, i32 } @f_f32x1_i32x1_s(float %0, i32 %1)
struct f32x1_i32x1_s f_f32x1_i32x1_s(struct f32x1_i32x1_s x) {
  return x;
}

/// 2. GRLEN < WOA ≤ 2 × GRLEN
/// a. Only fixed-point members.
/// i. The argument is passed in a pair of available GAR, with the low-order
/// bits in the lower-numbered GAR and the high-order bits in the
/// higher-numbered GAR. If only one GAR is available, the low-order bits are in
/// the GAR and the high-order bits are on the stack, and passed on the stack if
/// no GAR is available.

struct i64x2_s {
  int64_t a, b;
};

// CHECK-LABEL: define{{.*}} [2 x i64] @f_i64x2_s([2 x i64] %x.coerce)
struct i64x2_s f_i64x2_s(struct i64x2_s x) {
  return x;
}

/// b. Only floating-point members.
/// i. The structure has one long double member or one double member and two
/// adjacent float members or 3-4 float members. The argument is passed in a
/// pair of available GAR, with the low-order bits in the lower-numbered GAR and
/// the high-order bits in the higher-numbered GAR. If only one GAR is
/// available, the low-order bits are in the GAR and the high-order bits are on
/// the stack, and passed on the stack if no GAR is available.

struct f128x1_s {
  long double a;
};

// CHECK-LABEL: define{{.*}} i128 @f_f128x1_s(i128 %x.coerce)
struct f128x1_s f_f128x1_s(struct f128x1_s x) {
  return x;
}

struct f64x1_f32x2_s {
  double a;
  float b, c;
};

// CHECK-LABEL: define{{.*}} [2 x i64] @f_f64x1_f32x2_s([2 x i64] %x.coerce)
struct f64x1_f32x2_s f_f64x1_f32x2_s(struct f64x1_f32x2_s x) {
  return x;
}

struct f32x3_s {
  float a, b, c;
};

// CHECK-LABEL: define{{.*}} [2 x i64] @f_f32x3_s([2 x i64] %x.coerce)
struct f32x3_s f_f32x3_s(struct f32x3_s x) {
  return x;
}

struct f32x4_s {
  float a, b, c, d;
};

// CHECK-LABEL: define{{.*}} [2 x i64] @f_f32x4_s([2 x i64] %x.coerce)
struct f32x4_s f_f32x4_s(struct f32x4_s x) {
  return x;
}

/// ii. The structure with two double members is passed in a pair of available
/// FARs. If no a pair of available FARs, it’s passed in GARs. A structure with
/// one double member and one float member is same.

struct f64x2_s {
  double a, b;
};

// CHECK-LABEL: define{{.*}} { double, double } @f_f64x2_s(double %0, double %1)
struct f64x2_s f_f64x2_s(struct f64x2_s x) {
  return x;
}

/// c. Both fixed-point and floating-point members.
/// i. The structure has one double member and only one fixed-point member.
/// A. If one FAR and one GAR are available, the floating-point member of the
/// structure is passed in the FAR, and the integer member of the structure is
/// passed in the GAR; If no floating-point registers but two GARs are
/// available, it’s passed in the two GARs; If only one GAR is available, the
/// low-order bits are in the GAR and the high-order bits are on the stack; And
/// it’s passed on the stack if no GAR is available.

struct f64x1_i64x1_s {
  double a;
  int64_t b;
};

// CHECK-LABEL: define{{.*}} { double, i64 } @f_f64x1_i64x1_s(double %0, i64 %1)
struct f64x1_i64x1_s f_f64x1_i64x1_s(struct f64x1_i64x1_s x) {
  return x;
}

/// ii. Others
/// A. The argument is passed in a pair of available GAR, with the low-order
/// bits in the lower-numbered GAR and the high-order bits in the
/// higher-numbered GAR. If only one GAR is available, the low-order bits are in
/// the GAR and the high-order bits are on the stack, and passed on the stack if
/// no GAR is available.

struct f64x1_i32x2_s {
  double a;
  int32_t b, c;
};

// CHECK-LABEL: define{{.*}} [2 x i64] @f_f64x1_i32x2_s([2 x i64] %x.coerce)
struct f64x1_i32x2_s f_f64x1_i32x2_s(struct f64x1_i32x2_s x) {
  return x;
}

struct f32x2_i32x2_s {
  float a, b;
  int32_t c, d;
};

// CHECK-LABEL: define{{.*}} [2 x i64] @f_f32x2_i32x2_s([2 x i64] %x.coerce)
struct f32x2_i32x2_s f_f32x2_i32x2_s(struct f32x2_i32x2_s x) {
  return x;
}

/// 3. WOA > 2 × GRLEN
/// a. It’s passed by reference and are replaced in the argument list with the
/// address. If there is an available GAR, the reference is passed in the GAR,
/// and passed on the stack if no GAR is available.

struct i64x4_s {
  int64_t a, b, c, d;
};

// CHECK-LABEL: define{{.*}} void @f_i64x4_s(ptr{{.*}} sret(%struct.i64x4_s) align 8 %agg.result, ptr{{.*}} %x)
struct i64x4_s f_i64x4_s(struct i64x4_s x) {
  return x;
}

struct f64x4_s {
  double a, b, c, d;
};

// CHECK-LABEL: define{{.*}} void @f_f64x4_s(ptr{{.*}} sret(%struct.f64x4_s) align 8 %agg.result, ptr{{.*}} %x)
struct f64x4_s f_f64x4_s(struct f64x4_s x) {
  return x;
}

/// Part 3: Union arguments and return value.

/// Check empty unions are ignored.

union empty_u {};

// CHECK-LABEL: define{{.*}} void @f_empty_u()
union empty_u f_empty_u(union empty_u x) {
  return x;
}

/// Union is passed in GAR or stack.
/// 1. 0 < WOA ≤ GRLEN
/// a. The argument is passed in a GAR, or on the stack by value if no GAR is
/// available.

union i32_f32_u {
  int32_t a;
  float b;
};

// CHECK-LABEL: define{{.*}} i64 @f_i32_f32_u(i64 %x.coerce)
union i32_f32_u f_i32_f32_u(union i32_f32_u x) {
  return x;
}

union i64_f64_u {
  int64_t a;
  double b;
};

// CHECK-LABEL: define{{.*}} i64 @f_i64_f64_u(i64 %x.coerce)
union i64_f64_u f_i64_f64_u(union i64_f64_u x) {
  return x;
}

/// 2. GRLEN < WOA ≤ 2 × GRLEN
/// a. The argument is passed in a pair of available GAR, with the low-order
/// bits in the lower-numbered GAR and the high-order bits in the
/// higher-numbered GAR. If only one GAR is available, the low-order bits are in
/// the GAR and the high-order bits are on the stack. The arguments are passed
/// on the stack when no GAR is available.

union i128_f128_u {
  __int128_t a;
  long double b;
};

// CHECK-LABEL: define{{.*}} i128 @f_i128_f128_u(i128 %x.coerce)
union i128_f128_u f_i128_f128_u(union i128_f128_u x) {
  return x;
}

/// 3. WOA > 2 × GRLEN
/// a. It’s passed by reference and are replaced in the argument list with the
/// address. If there is an available GAR, the reference is passed in the GAR,
/// and passed on the stack if no GAR is available.

union i64_arr3_u {
  int64_t a[3];
};

// CHECK-LABEL: define{{.*}} void @f_i64_arr3_u(ptr{{.*}} sret(%union.i64_arr3_u) align 8 %agg.result, ptr{{.*}} %x)
union i64_arr3_u f_i64_arr3_u(union i64_arr3_u x) {
  return x;
}

/// Part 4: Complex number arguments and return value.

/// A complex floating-point number, or a structure containing just one complex
/// floating-point number, is passed as though it were a structure containing
/// two floating-point reals.

// CHECK-LABEL: define{{.*}} { float, float } @f_floatcomplex(float noundef %x.coerce0, float noundef %x.coerce1)
float __complex__ f_floatcomplex(float __complex__ x) { return x; }

// CHECK-LABEL: define{{.*}} { double, double } @f_doublecomplex(double noundef %x.coerce0, double noundef %x.coerce1)
double __complex__ f_doublecomplex(double __complex__ x) { return x; }

struct floatcomplex_s {
  float __complex__ c;
};
// CHECK-LABEL: define{{.*}} { float, float } @f_floatcomplex_s(float %0, float %1)
struct floatcomplex_s f_floatcomplex_s(struct floatcomplex_s x) {
  return x;
}

struct doublecomplex_s {
  double __complex__ c;
};
// CHECK-LABEL: define{{.*}} { double, double } @f_doublecomplex_s(double %0, double %1)
struct doublecomplex_s f_doublecomplex_s(struct doublecomplex_s x) {
  return x;
}

/// Part 5: Variadic arguments.

/// Variadic arguments are passed in GARs in the same manner as named arguments.

int f_va_callee(int, ...);

// CHECK-LABEL: define{{.*}} void @f_va_caller()
// CHECK: call signext i32 (i32, ...) @f_va_callee(i32 noundef signext 1, i32 noundef signext 2, i64 noundef 3, double noundef 4.000000e+00, double noundef 5.000000e+00, i64 {{.*}}, [2 x i64] {{.*}})
void f_va_caller(void) {
  f_va_callee(1, 2, 3LL, 4.0f, 5.0, (struct i16x4_s){6, 7, 8, 9},
              (struct i64x2_s){10, 11});
}

// CHECK-LABEL: @f_va_int(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FMT_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[VA:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[V:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store ptr [[FMT:%.*]], ptr [[FMT_ADDR]], align 8
// CHECK-NEXT:    call void @llvm.va_start.p0(ptr [[VA]])
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[VA]], align 8
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i64 8
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[VA]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[ARGP_CUR]], align 8
// CHECK-NEXT:    store i32 [[TMP0]], ptr [[V]], align 4
// CHECK-NEXT:    call void @llvm.va_end.p0(ptr [[VA]])
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[V]], align 4
// CHECK-NEXT:    ret i32 [[TMP1]]
int f_va_int(char *fmt, ...) {
  __builtin_va_list va;
  __builtin_va_start(va, fmt);
  int v = __builtin_va_arg(va, int);
  __builtin_va_end(va);
  return v;
}

/// Part 6. Structures with zero size fields (bitfields or arrays).

/// Check that zero size fields in structure are ignored.
/// Note that this rule is not explicitly documented in ABI spec but it matches
/// GCC's behavior.

struct f64x2_zsfs_s {
  double a;
  int : 0;
  __int128_t : 0;
  int b[0];
  __int128_t c[0];
  double d;
};

// CHECK-LABEL: define{{.*}} { double, double } @f_f64x2_zsfs_s(double %0, double %1)
struct f64x2_zsfs_s f_f64x2_zsfs_s(struct f64x2_zsfs_s x) {
  return x;
}

