// RUN: %clang_cc1 -triple sparc-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=V8
// RUN: %clang_cc1 -triple sparcv9-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=V9

// RUN: %clang_cc1 -triple sparc-unknown-linux-gnu -fclang-abi-compat=23 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=COMPAT23-V8
// RUN: %clang_cc1 -triple sparcv9-unknown-linux-gnu -fclang-abi-compat=23 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=COMPAT23-V9

// Test how SPARC passes and returns `_Complex` values.

// Returns.
//
// A `_Complex` value with an integer element type is returned packed into whole
// integer registers. Clang 23 and before instead gave each part a register of
// its own on v8, and on v9 left-justified a value narrower than a register the
// way a small struct is returned. The new behavior matches GCC.

// COMPAT23-V8-LABEL: define{{.*}} { i8, i8 } @ret_complex_char(
// V8-LABEL:          define{{.*}} i16 @ret_complex_char(
// COMPAT23-V9-LABEL: define{{.*}} i64 @ret_complex_char(
// V9-LABEL:          define{{.*}} i16 @ret_complex_char(
_Complex char ret_complex_char(void) { return 0; }

// COMPAT23-V8-LABEL: define{{.*}} { i16, i16 } @ret_complex_short(
// V8-LABEL:          define{{.*}} i32 @ret_complex_short(
// COMPAT23-V9-LABEL: define{{.*}} i64 @ret_complex_short(
// V9-LABEL:          define{{.*}} i32 @ret_complex_short(
_Complex short ret_complex_short(void) { return 0; }

// COMPAT23-V8-LABEL: define{{.*}} { i32, i32 } @ret_complex_int(
// V8-LABEL:          define{{.*}} i64 @ret_complex_int(
// COMPAT23-V9-LABEL: define{{.*}} i64 @ret_complex_int(
// V9-LABEL:          define{{.*}} i64 @ret_complex_int(
_Complex int ret_complex_int(void) { return 0; }

// COMPAT23-V8-LABEL: define{{.*}} { i64, i64 } @ret_complex_long_long(
// V8-LABEL:          define{{.*}} { i64, i64 } @ret_complex_long_long(
// COMPAT23-V9-LABEL: define{{.*}} { i64, i64 } @ret_complex_long_long(
// V9-LABEL:          define{{.*}} { i64, i64 } @ret_complex_long_long(
_Complex long long ret_complex_long_long(void) { return 0; }

// COMPAT23-V8-LABEL: define{{.*}} { float, float } @ret_complex_float(
// V8-LABEL:          define{{.*}} { float, float } @ret_complex_float(
// COMPAT23-V9-LABEL: define{{.*}} inreg { float, float } @ret_complex_float(
// V9-LABEL:          define{{.*}} inreg { float, float } @ret_complex_float(
_Complex float ret_complex_float(void) { return 0; }

// COMPAT23-V8-LABEL: define{{.*}} { double, double } @ret_complex_double(
// V8-LABEL:          define{{.*}} { double, double } @ret_complex_double(
// COMPAT23-V9-LABEL: define{{.*}} { double, double } @ret_complex_double(
// V9-LABEL:          define{{.*}} { double, double } @ret_complex_double(
_Complex double ret_complex_double(void) { return 0; }

// COMPAT23-V8-LABEL: define{{.*}} inreg { fp128, fp128 } @ret_complex_long_double(
// V8-LABEL:          define{{.*}} inreg { fp128, fp128 } @ret_complex_long_double(
// COMPAT23-V9-LABEL: define{{.*}} { fp128, fp128 } @ret_complex_long_double(
// V9-LABEL:          define{{.*}} { fp128, fp128 } @ret_complex_long_double(
_Complex long double ret_complex_long_double(void) { return 0; }

// Arguments.

// COMPAT23-V8-LABEL: define{{.*}} void @arg_complex_char(ptr noundef byval({ i8, i8 }) align 1 %c)
// V8-LABEL:          define{{.*}} void @arg_complex_char(i16 noundef %c.coerce)
// COMPAT23-V9-LABEL: define{{.*}} void @arg_complex_char(i64 %c.coerce)
// V9-LABEL:          define{{.*}} void @arg_complex_char(i16 noundef %c.coerce)
void arg_complex_char(_Complex char c) {}

// COMPAT23-V8-LABEL: define{{.*}} void @arg_complex_short(ptr noundef byval({ i16, i16 }) align 2 %c)
// V8-LABEL:          define{{.*}} void @arg_complex_short(i32 noundef %c.coerce)
// COMPAT23-V9-LABEL: define{{.*}} void @arg_complex_short(i64 %c.coerce)
// V9-LABEL:          define{{.*}} void @arg_complex_short(i32 noundef %c.coerce)
void arg_complex_short(_Complex short c) {}

// COMPAT23-V8-LABEL: define{{.*}} void @arg_complex_int(ptr noundef byval({ i32, i32 }) align 4 %c)
// V8-LABEL:          define{{.*}} void @arg_complex_int(i64 noundef %c.coerce)
// COMPAT23-V9-LABEL: define{{.*}} void @arg_complex_int(i64 noundef %c.coerce)
// V9-LABEL:          define{{.*}} void @arg_complex_int(i64 noundef %c.coerce)
void arg_complex_int(_Complex int c) {}

// COMPAT23-V8-LABEL: define{{.*}} void @arg_complex_long_long(ptr noundef byval({ i64, i64 }) align 8 %c)
// V8-LABEL:          define{{.*}} void @arg_complex_long_long(ptr noundef byval({ i64, i64 }) align 8 %c)
// COMPAT23-V9-LABEL: define{{.*}} void @arg_complex_long_long(i64 noundef %c.coerce0, i64 noundef %c.coerce1)
// V9-LABEL:          define{{.*}} void @arg_complex_long_long(i64 noundef %c.coerce0, i64 noundef %c.coerce1)
void arg_complex_long_long(_Complex long long c) {}

// COMPAT23-V8-LABEL: define{{.*}} void @arg_complex_float(ptr noundef byval({ float, float }) align 4 %c)
// V8-LABEL:          define{{.*}} void @arg_complex_float(ptr noundef byval({ float, float }) align 4 %c)
// COMPAT23-V9-LABEL: define{{.*}} void @arg_complex_float(float inreg noundef %c.coerce0, float inreg noundef %c.coerce1)
// V9-LABEL:          define{{.*}} void @arg_complex_float(float inreg noundef %c.coerce0, float inreg noundef %c.coerce1)
void arg_complex_float(_Complex float c) {}

// COMPAT23-V8-LABEL: define{{.*}} void @arg_complex_double(ptr noundef byval({ double, double }) align 8 %c)
// V8-LABEL:          define{{.*}} void @arg_complex_double(ptr noundef byval({ double, double }) align 8 %c)
// COMPAT23-V9-LABEL: define{{.*}} void @arg_complex_double(double noundef %c.coerce0, double noundef %c.coerce1)
// V9-LABEL:          define{{.*}} void @arg_complex_double(double noundef %c.coerce0, double noundef %c.coerce1)
void arg_complex_double(_Complex double c) {}

// COMPAT23-V8-LABEL: define{{.*}} void @arg_complex_long_double(ptr noundef byval({ fp128, fp128 }) align 8 %c)
// V8-LABEL:          define{{.*}} void @arg_complex_long_double(ptr noundef byval({ fp128, fp128 }) align 8 %c)
// COMPAT23-V9-LABEL: define{{.*}} void @arg_complex_long_double(ptr noundef align 16 dead_on_return %c)
// V9-LABEL:          define{{.*}} void @arg_complex_long_double(ptr noundef align 16 dead_on_return %c)
void arg_complex_long_double(_Complex long double c) {}
