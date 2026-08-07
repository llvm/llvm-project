// RUN: %clang_cc1 -triple mips-none-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=GPR,GPR32
// RUN: %clang_cc1 -triple mipsel-none-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=GPR,GPR32

// RUN: %clang_cc1 -triple mips64-none-linux-gnu -target-abi n32 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=GPR,GPR64
// RUN: %clang_cc1 -triple mips64-none-linux-gnu -target-abi n64 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=GPR,GPR64
// RUN: %clang_cc1 -triple mips64el-none-linux-gnu -target-abi n64 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=GPR,GPR64

// RUN: %clang_cc1 -triple mips-none-linux-gnu -fclang-abi-compat=23 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=COMPAT23,COMPAT23-GPR32
// RUN: %clang_cc1 -triple mips64-none-linux-gnu -target-abi n64 -fclang-abi-compat=23 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=COMPAT23,COMPAT23-GPR64

// Test how MIPS passes and returns `_Complex` values.

// With Clang 23 and before, `_Complex {integer}` was returned in 2 registers.
// Later versions are compatible with GCC and pack such types into a single
// GPR if possible.

// GPR-LABEL:            define{{.*}} i16 @ret_complex_char(
// COMPAT23-LABEL:       define{{.*}} { i8, i8 } @ret_complex_char(
_Complex char ret_complex_char(void) { return 0; }

// GPR-LABEL:            define{{.*}} i32 @ret_complex_short(
// COMPAT23-LABEL:       define{{.*}} { i16, i16 } @ret_complex_short(
_Complex short ret_complex_short(void) { return 0; }

// GPR32-LABEL:          define{{.*}} { i32, i32 } @ret_complex_int(
// GPR64-LABEL:          define{{.*}} i64 @ret_complex_int(
// COMPAT23-LABEL:       define{{.*}} { i32, i32 } @ret_complex_int(
_Complex int ret_complex_int(void) { return 0; }

// GPR-LABEL:            define{{.*}} { i64, i64 } @ret_complex_long_long(
// COMPAT23-LABEL:       define{{.*}} { i64, i64 } @ret_complex_long_long(
_Complex long long ret_complex_long_long(void) { return 0; }

// A `_Complex` value with a floating-point element type is returned in FPRs.

// GPR-LABEL:            define{{.*}} { float, float } @ret_complex_float(
// COMPAT23-LABEL:       define{{.*}} { float, float } @ret_complex_float(
_Complex float ret_complex_float(void) { return 0; }

// GPR-LABEL:            define{{.*}} { double, double } @ret_complex_double(
// COMPAT23-LABEL:       define{{.*}} { double, double } @ret_complex_double(
_Complex double ret_complex_double(void) { return 0; }

// GPR32-LABEL:          define{{.*}} { double, double } @ret_complex_long_double(
// COMPAT23-GPR32-LABEL: define{{.*}} { double, double } @ret_complex_long_double(
// GPR64-LABEL:          define{{.*}} void @ret_complex_long_double(ptr {{.*}}sret({ fp128, fp128 })
// COMPAT23-GPR64-LABEL: define{{.*}} void @ret_complex_long_double(ptr {{.*}}sret({ fp128, fp128 })
_Complex long double ret_complex_long_double(void) { return 0; }

// Arguments

// GPR-LABEL:            define{{.*}} void @arg_complex_char(i16 inreg noundef %c.coerce)
void arg_complex_char(_Complex char c) {}

// GPR-LABEL:            define{{.*}} void @arg_complex_short(i32 inreg noundef %c.coerce)
void arg_complex_short(_Complex short c) {}

// GPR32-LABEL:          define{{.*}} void @arg_complex_int(i32 inreg noundef %c.coerce0, i32 inreg noundef %c.coerce1)
// GPR64-LABEL:          define{{.*}} void @arg_complex_int(i64 inreg noundef %c.coerce)
void arg_complex_int(_Complex int c) {}

// GPR32-LABEL:          define{{.*}} void @arg_complex_long_long(i32 inreg noundef %c.coerce0, i32 inreg noundef %c.coerce1, i32 inreg noundef %c.coerce2, i32 inreg noundef %c.coerce3)
// GPR64-LABEL:          define{{.*}} void @arg_complex_long_long(i64 inreg noundef %c.coerce0, i64 inreg noundef %c.coerce1)
void arg_complex_long_long(_Complex long long c) {}

// GPR32-LABEL:          define{{.*}} void @arg_complex_float(i32 inreg noundef %c.coerce0, i32 inreg noundef %c.coerce1)
// GPR64-LABEL:          define{{.*}} void @arg_complex_float(float inreg noundef %c.coerce0, float inreg noundef %c.coerce1)
void arg_complex_float(_Complex float c) {}

// GPR32-LABEL:          define{{.*}} void @arg_complex_double(i32 inreg noundef %c.coerce0, i32 inreg noundef %c.coerce1, i32 inreg noundef %c.coerce2, i32 inreg noundef %c.coerce3)
// GPR64-LABEL:          define{{.*}} void @arg_complex_double(double inreg noundef %c.coerce0, double inreg noundef %c.coerce1)
void arg_complex_double(_Complex double c) {}

// GPR32-LABEL:          define{{.*}} void @arg_complex_long_double(i32 inreg noundef %c.coerce0, i32 inreg noundef %c.coerce1, i32 inreg noundef %c.coerce2, i32 inreg noundef %c.coerce3)
// GPR64-LABEL:          define{{.*}} void @arg_complex_long_double(fp128 inreg noundef %c.coerce0, fp128 inreg noundef %c.coerce1)
void arg_complex_long_double(_Complex long double c) {}

// Straddling the FPR/GPR border

// With Clang 23 and before, `_Complex float` and `_Complex` double were passed as two floats or doubles, 
// even when that would not fit in the remaining float registers. Later versions match GCC, which will 
// cast to one (float) or two (double) i64 values which are passed via GPRs (or the stack).

// Just padding to fill the slots.
#define SIX_SLOTS  long long a0, long long a1, long long a2, \
                   long long a3, long long a4, long long a5
#define SEVEN_SLOTS SIX_SLOTS, long long a6

// Six slots used, so two are still free and the FPR pair is still used.

// GPR32-LABEL:          define{{.*}} @arg_complex_float_6slots(i64{{.*}}, i32 inreg noundef %c.coerce0, i32 inreg noundef %c.coerce1)
// GPR64-LABEL:          define{{.*}} @arg_complex_float_6slots(i64{{.*}}, float inreg noundef %c.coerce0, float inreg noundef %c.coerce1)
// COMPAT23-GPR64-LABEL: define{{.*}} @arg_complex_float_6slots(i64{{.*}}, float inreg noundef %c.coerce0, float inreg noundef %c.coerce1)
void arg_complex_float_6slots(SIX_SLOTS, _Complex float c) {}

// GPR32-LABEL:          define{{.*}} @arg_complex_double_6slots(i64{{.*}}, i32 inreg noundef %c.coerce0, i32 inreg noundef %c.coerce1, i32 inreg noundef %c.coerce2, i32 inreg noundef %c.coerce3)
// GPR64-LABEL:          define{{.*}} @arg_complex_double_6slots(i64{{.*}}, double inreg noundef %c.coerce0, double inreg noundef %c.coerce1)
// COMPAT23-GPR64-LABEL: define{{.*}} @arg_complex_double_6slots(i64{{.*}}, double inreg noundef %c.coerce0, double inreg noundef %c.coerce1)
void arg_complex_double_6slots(SIX_SLOTS, _Complex double c) {}

// GPR32-LABEL:          define{{.*}} @arg_complex_float_7slots(i64{{.*}}, i32 inreg noundef %c.coerce0, i32 inreg noundef %c.coerce1)
// GPR64-LABEL:          define{{.*}} @arg_complex_float_7slots(i64{{.*}}, i64 inreg noundef %c.coerce)
// COMPAT23-GPR64-LABEL: define{{.*}} @arg_complex_float_7slots(i64{{.*}}, float inreg noundef %c.coerce0, float inreg noundef %c.coerce1)
void arg_complex_float_7slots(SEVEN_SLOTS, _Complex float c) {}

// GPR32-LABEL:          define{{.*}} @arg_complex_double_7slots(i64{{.*}}, i32 inreg noundef %c.coerce0, i32 inreg noundef %c.coerce1, i32 inreg noundef %c.coerce2, i32 inreg noundef %c.coerce3)
// GPR64-LABEL:          define{{.*}} @arg_complex_double_7slots(i64{{.*}}, i64 inreg noundef %c.coerce0, i64 inreg noundef %c.coerce1)
// COMPAT23-GPR64-LABEL: define{{.*}} @arg_complex_double_7slots(i64{{.*}}, double inreg noundef %c.coerce0, double inreg noundef %c.coerce1)
void arg_complex_double_7slots(SEVEN_SLOTS, _Complex double c) {}

// GPR64-LABEL:          define{{.*}} @arg_complex_float_after_3({{.*}}, float inreg noundef %c.coerce0, float inreg noundef %c.coerce1)
void arg_complex_float_after_3(_Complex float p0, _Complex float p1, _Complex float p2, _Complex float c) {}

// GPR64-LABEL:          define{{.*}} @arg_complex_float_after_4({{.*}}, i64 inreg noundef %c.coerce)
void arg_complex_float_after_4(_Complex float p0, _Complex float p1, _Complex float p2, _Complex float p3, _Complex float c) {}

// GPR64-LABEL:          define{{.*}} @arg_complex_long_double_7slots(i64{{.*}}, fp128 inreg noundef %c.coerce0, fp128 inreg noundef %c.coerce1)
// COMPAT23-GPR64-LABEL: define{{.*}} @arg_complex_long_double_7slots(i64{{.*}}, fp128 inreg noundef %c.coerce0, fp128 inreg noundef %c.coerce1)
void arg_complex_long_double_7slots(SEVEN_SLOTS, _Complex long double c) {}
