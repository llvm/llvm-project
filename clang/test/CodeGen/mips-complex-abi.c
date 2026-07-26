// RUN: %clang_cc1 -triple mips-none-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=GPR32
// RUN: %clang_cc1 -triple mipsel-none-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=GPR32

// RUN: %clang_cc1 -triple mips64-none-linux-gnu -target-abi n32 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=GPR64
// RUN: %clang_cc1 -triple mips64-none-linux-gnu -target-abi n64 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=GPR64
// RUN: %clang_cc1 -triple mips64el-none-linux-gnu -target-abi n64 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=GPR64

// RUN: %clang_cc1 -triple mips-none-linux-gnu -fclang-abi-compat=23 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=COMPAT23
// RUN: %clang_cc1 -triple mips64-none-linux-gnu -target-abi n64 -fclang-abi-compat=23 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=COMPAT23

// Test how MIPS passes and returns `_Complex` values.

// With Clang 23 and before, `_Complex {integer}` was passed in 2 registers.
// Later versions are compatible with GCC and pack such types into a single
// GPR if possible.

// GPR32-LABEL:    define{{.*}} i16 @ret_complex_char(
// GPR64-LABEL:    define{{.*}} i16 @ret_complex_char(
// COMPAT23-LABEL: define{{.*}} { i8, i8 } @ret_complex_char(
_Complex char ret_complex_char(void) { return 0; }

// GPR32-LABEL:    define{{.*}} i32 @ret_complex_short(
// GPR64-LABEL:    define{{.*}} i32 @ret_complex_short(
// COMPAT23-LABEL: define{{.*}} { i16, i16 } @ret_complex_short(
_Complex short ret_complex_short(void) { return 0; }

// GPR32-LABEL:    define{{.*}} { i32, i32 } @ret_complex_int(
// GPR64-LABEL:    define{{.*}} i64 @ret_complex_int(
// COMPAT23-LABEL: define{{.*}} { i32, i32 } @ret_complex_int(
_Complex int ret_complex_int(void) { return 0; }

// GPR32-LABEL:    define{{.*}} { i64, i64 } @ret_complex_long_long(
// GPR64-LABEL:    define{{.*}} { i64, i64 } @ret_complex_long_long(
// COMPAT23-LABEL: define{{.*}} { i64, i64 } @ret_complex_long_long(
_Complex long long ret_complex_long_long(void) { return 0; }

// A `_Complex` value with a floating-point element type is returned in FPRs.

// GPR32-LABEL:    define{{.*}} { float, float } @ret_complex_float(
// GPR64-LABEL:    define{{.*}} { float, float } @ret_complex_float(
// COMPAT23-LABEL: define{{.*}} { float, float } @ret_complex_float(
_Complex float ret_complex_float(void) { return 0; }

// GPR32-LABEL:    define{{.*}} { double, double } @ret_complex_double(
// GPR64-LABEL:    define{{.*}} { double, double } @ret_complex_double(
// COMPAT23-LABEL: define{{.*}} { double, double } @ret_complex_double(
_Complex double ret_complex_double(void) { return 0; }

// Arguments

// GPR32-LABEL: define{{.*}} void @arg_complex_char(i16 inreg noundef %c.coerce)
// GPR64-LABEL: define{{.*}} void @arg_complex_char(i16 inreg noundef %c.coerce)
void arg_complex_char(_Complex char c) {}

// GPR32-LABEL: define{{.*}} void @arg_complex_short(i32 inreg noundef %c.coerce)
// GPR64-LABEL: define{{.*}} void @arg_complex_short(i32 inreg noundef %c.coerce)
void arg_complex_short(_Complex short c) {}

// GPR32-LABEL: define{{.*}} void @arg_complex_int(i32 inreg noundef %c.coerce0, i32 inreg noundef %c.coerce1)
// GPR64-LABEL: define{{.*}} void @arg_complex_int(i64 inreg noundef %c.coerce)
void arg_complex_int(_Complex int c) {}

// GPR32-LABEL: define{{.*}} void @arg_complex_long_long(i32 inreg noundef %c.coerce0, i32 inreg noundef %c.coerce1, i32 inreg noundef %c.coerce2, i32 inreg noundef %c.coerce3)
// GPR64-LABEL: define{{.*}} void @arg_complex_long_long(i64 inreg noundef %c.coerce0, i64 inreg noundef %c.coerce1)
void arg_complex_long_long(_Complex long long c) {}

// GPR32-LABEL: define{{.*}} void @arg_complex_float(i32 inreg noundef %c.coerce0, i32 inreg noundef %c.coerce1)
// GPR64-LABEL: define{{.*}} void @arg_complex_float(float inreg noundef %c.coerce0, float inreg noundef %c.coerce1)
void arg_complex_float(_Complex float c) {}

// GPR32-LABEL: define{{.*}} void @arg_complex_double(i32 inreg noundef %c.coerce0, i32 inreg noundef %c.coerce1, i32 inreg noundef %c.coerce2, i32 inreg noundef %c.coerce3)
// GPR64-LABEL: define{{.*}} void @arg_complex_double(double inreg noundef %c.coerce0, double inreg noundef %c.coerce1)
void arg_complex_double(_Complex double c) {}
