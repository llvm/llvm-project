// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -target-feature +float128 -DTEST_F128 -triple \
// RUN:   powerpc64le-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s \
// RUN:   --check-prefix CHECK-F128

float crealf(_Complex float);
double creal(_Complex double);
long double creall(_Complex long double);
#ifdef TEST_F128
__float128 crealf128(_Complex __float128);
_Complex __float128 foo_f128(_Complex __float128 x) {
  return x;
}

// CHECK-F128: define{{.*}} { fp128, fp128 } @foo_f128(fp128 {{[%A-Za-z0-9.]+}}, fp128 {{[%A-Za-z0-9.]+}}) [[NUW:#[0-9]+]] {
#endif

_Complex float foo_float(_Complex float x) {
  return x;
}

// CHECK: define{{.*}} { float, float } @foo_float(float {{[%A-Za-z0-9.]+}}, float {{[%A-Za-z0-9.]+}}) [[NUW:#[0-9]+]] {

_Complex double foo_double(_Complex double x) {
  return x;
}

// CHECK: define{{.*}} { double, double } @foo_double(double {{[%A-Za-z0-9.]+}}, double {{[%A-Za-z0-9.]+}}) [[NUW]] {

_Complex long double foo_long_double(_Complex long double x) {
  return x;
}

// CHECK: define{{.*}} { ppc_fp128, ppc_fp128 } @foo_long_double(ppc_fp128 {{[%A-Za-z0-9.]+}}, ppc_fp128 {{[%A-Za-z0-9.]+}}) [[NUW]] {

_Complex int foo_int(_Complex int x) {
  return x;
}

// CHECK: define{{.*}} { i32, i32 } @foo_int(i32 {{[%A-Za-z0-9.]+}}, i32 {{[%A-Za-z0-9.]+}}) [[NUW]] {

_Complex short foo_short(_Complex short x) {
  return x;
}

// CHECK: define{{.*}} { i16, i16 } @foo_short(i16 {{[%A-Za-z0-9.]+}}, i16 {{[%A-Za-z0-9.]+}}) [[NUW]] {

_Complex signed char foo_char(_Complex signed char x) {
  return x;
}

// CHECK: define{{.*}} { i8, i8 } @foo_char(i8 {{[%A-Za-z0-9.]+}}, i8 {{[%A-Za-z0-9.]+}}) [[NUW]] {

_Complex long foo_long(_Complex long x) {
  return x;
}

// CHECK: define{{.*}} { i64, i64 } @foo_long(i64 {{[%A-Za-z0-9.]+}}, i64 {{[%A-Za-z0-9.]+}}) [[NUW]] {

_Complex long long foo_long_long(_Complex long long x) {
  return x;
}

// CHECK: define{{.*}} { i64, i64 } @foo_long_long(i64 {{[%A-Za-z0-9.]+}}, i64 {{[%A-Za-z0-9.]+}}) [[NUW]] {

float bar_float(void) {
  return crealf(foo_float(2.0f - 2.5fi));
}

// CHECK: define{{.*}} float @bar_float() [[NUW]] {
// CHECK: [[VAR1:[%A-Za-z0-9.]+]] = call { float, float } @foo_float
// CHECK: extractvalue { float, float } [[VAR1]], 0
// CHECK: extractvalue { float, float } [[VAR1]], 1

double bar_double(void) {
  return creal(foo_double(2.0 - 2.5i));
}

// CHECK: define{{.*}} double @bar_double() [[NUW]] {
// CHECK: [[VAR2:[%A-Za-z0-9.]+]] = call { double, double } @foo_double
// CHECK: extractvalue { double, double } [[VAR2]], 0
// CHECK: extractvalue { double, double } [[VAR2]], 1

long double bar_long_double(void) {
  return creall(foo_long_double(2.0L - 2.5Li));
}

// CHECK: define{{.*}} ppc_fp128 @bar_long_double() [[NUW]] {
// CHECK: [[VAR3:[%A-Za-z0-9.]+]] = call { ppc_fp128, ppc_fp128 } @foo_long_double
// CHECK: extractvalue { ppc_fp128, ppc_fp128 } [[VAR3]], 0
// CHECK: extractvalue { ppc_fp128, ppc_fp128 } [[VAR3]], 1

#ifdef TEST_F128
__float128 bar_f128(void) {
  return crealf128(foo_f128(2.0Q - 2.5Qi));
}

// CHECK-F128: define{{.*}} fp128 @bar_f128() [[NUW]] {
// CHECK-F128: [[VAR3:[%A-Za-z0-9.]+]] = call { fp128, fp128 } @foo_f128
// CHECK-F128: extractvalue { fp128, fp128 } [[VAR3]], 0
// CHECK-F128: extractvalue { fp128, fp128 } [[VAR3]], 1
#endif

int bar_int(void) {
  return __real__(foo_int(2 - 3i));
}

// CHECK: define{{.*}} signext i32 @bar_int() [[NUW]] {
// CHECK: [[VAR4:[%A-Za-z0-9.]+]] = call { i32, i32 } @foo_int
// CHECK: extractvalue { i32, i32 } [[VAR4]], 0
// CHECK: extractvalue { i32, i32 } [[VAR4]], 1

short bar_short(void) {
  return __real__(foo_short(2 - 3i));
}

// CHECK: define{{.*}} signext i16 @bar_short() [[NUW]] {
// CHECK: [[VAR5:[%A-Za-z0-9.]+]] = call { i16, i16 } @foo_short
// CHECK: extractvalue { i16, i16 } [[VAR5]], 0
// CHECK: extractvalue { i16, i16 } [[VAR5]], 1

signed char bar_char(void) {
  return __real__(foo_char(2 - 3i));
}

// CHECK: define{{.*}} signext i8 @bar_char() [[NUW]] {
// CHECK: [[VAR6:[%A-Za-z0-9.]+]] = call { i8, i8 } @foo_char
// CHECK: extractvalue { i8, i8 } [[VAR6]], 0
// CHECK: extractvalue { i8, i8 } [[VAR6]], 1

long bar_long(void) {
  return __real__(foo_long(2L - 3Li));
}

// CHECK: define{{.*}} i64 @bar_long() [[NUW]] {
// CHECK: [[VAR7:[%A-Za-z0-9.]+]] = call { i64, i64 } @foo_long
// CHECK: extractvalue { i64, i64 } [[VAR7]], 0
// CHECK: extractvalue { i64, i64 } [[VAR7]], 1

long long bar_long_long(void) {
  return __real__(foo_long_long(2LL - 3LLi));
}

// CHECK: define{{.*}} i64 @bar_long_long() [[NUW]] {
// CHECK: [[VAR8:[%A-Za-z0-9.]+]] = call { i64, i64 } @foo_long_long
// CHECK: extractvalue { i64, i64 } [[VAR8]], 0
// CHECK: extractvalue { i64, i64 } [[VAR8]], 1


// CHECK: attributes [[NUW]] = { noinline nounwind{{.*}} }
