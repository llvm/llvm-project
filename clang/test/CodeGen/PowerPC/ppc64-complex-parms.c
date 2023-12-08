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
__float128 foo_f128(_Complex __float128 x) {
  return crealf128(x);
}
// CHECK-F128: define{{.*}} fp128 @foo_f128(fp128 noundef {{[%A-Za-z0-9.]+}}, fp128 noundef {{[%A-Za-z0-9.]+}})
#endif

float foo_float(_Complex float x) {
  return crealf(x);
}

// CHECK: define{{.*}} float @foo_float(float noundef {{[%A-Za-z0-9.]+}}, float noundef {{[%A-Za-z0-9.]+}}) [[NUW:#[0-9]+]] {

double foo_double(_Complex double x) {
  return creal(x);
}

// CHECK: define{{.*}} double @foo_double(double noundef {{[%A-Za-z0-9.]+}}, double noundef {{[%A-Za-z0-9.]+}}) [[NUW]] {

long double foo_long_double(_Complex long double x) {
  return creall(x);
}

// CHECK: define{{.*}} ppc_fp128 @foo_long_double(ppc_fp128 noundef {{[%A-Za-z0-9.]+}}, ppc_fp128 noundef {{[%A-Za-z0-9.]+}}) [[NUW]] {

__ibm128 foo_ibm128(_Complex __ibm128 x) {
  // We don't have a suffix for explicit __ibm128 type yet. Use *l instead.
  return creall(x);
}

// CHECK: define{{.*}} ppc_fp128 @foo_ibm128(ppc_fp128 noundef {{[%A-Za-z0-9.]+}}, ppc_fp128 noundef {{[%A-Za-z0-9.]+}}) [[NUW]] {

int foo_int(_Complex int x) {
  return __real__ x;
}

// CHECK: define{{.*}} signext i32 @foo_int(i32 noundef {{[%A-Za-z0-9.]+}}, i32 noundef {{[%A-Za-z0-9.]+}}) [[NUW]] {

short foo_short(_Complex short x) {
  return __real__ x;
}

// CHECK: define{{.*}} signext i16 @foo_short(i16 noundef {{[%A-Za-z0-9.]+}}, i16 noundef {{[%A-Za-z0-9.]+}}) [[NUW]] {

signed char foo_char(_Complex signed char x) {
  return __real__ x;
}

// CHECK: define{{.*}} signext i8 @foo_char(i8 noundef {{[%A-Za-z0-9.]+}}, i8 noundef {{[%A-Za-z0-9.]+}}) [[NUW]] {

long foo_long(_Complex long x) {
  return __real__ x;
}

// CHECK: define{{.*}} i64 @foo_long(i64 noundef {{[%A-Za-z0-9.]+}}, i64 noundef {{[%A-Za-z0-9.]+}}) [[NUW]] {

long long foo_long_long(_Complex long long x) {
  return __real__ x;
}

// CHECK: define{{.*}} i64 @foo_long_long(i64 noundef {{[%A-Za-z0-9.]+}}, i64 noundef {{[%A-Za-z0-9.]+}}) [[NUW]] {

void bar_float(void) {
  foo_float(2.0f - 2.5fi);
}

// CHECK: define{{.*}} void @bar_float() [[NUW]] {
// CHECK: %[[VAR1:[A-Za-z0-9.]+]] = alloca { float, float }, align 4
// CHECK: %[[VAR2:[A-Za-z0-9.]+]] = getelementptr inbounds { float, float }, ptr %[[VAR1]], i32 0, i32 0
// CHECK: %[[VAR3:[A-Za-z0-9.]+]] = getelementptr inbounds { float, float }, ptr %[[VAR1]], i32 0, i32 1
// CHECK: store float 2.000000e+00, ptr %[[VAR2]]
// CHECK: store float -2.500000e+00, ptr %[[VAR3]]
// CHECK: %[[VAR4:[A-Za-z0-9.]+]] = getelementptr inbounds { float, float }, ptr %[[VAR1]], i32 0, i32 0
// CHECK: %[[VAR5:[A-Za-z0-9.]+]] = load float, ptr %[[VAR4]], align 4
// CHECK: %[[VAR6:[A-Za-z0-9.]+]] = getelementptr inbounds { float, float }, ptr %[[VAR1]], i32 0, i32 1
// CHECK: %[[VAR7:[A-Za-z0-9.]+]] = load float, ptr %[[VAR6]], align 4
// CHECK: %{{[A-Za-z0-9.]+}} = call float @foo_float(float noundef %[[VAR5]], float noundef %[[VAR7]])

void bar_double(void) {
  foo_double(2.0 - 2.5i);
}

// CHECK: define{{.*}} void @bar_double() [[NUW]] {
// CHECK: %[[VAR11:[A-Za-z0-9.]+]] = alloca { double, double }, align 8
// CHECK: %[[VAR12:[A-Za-z0-9.]+]] = getelementptr inbounds { double, double }, ptr %[[VAR11]], i32 0, i32 0
// CHECK: %[[VAR13:[A-Za-z0-9.]+]] = getelementptr inbounds { double, double }, ptr %[[VAR11]], i32 0, i32 1
// CHECK: store double 2.000000e+00, ptr %[[VAR12]]
// CHECK: store double -2.500000e+00, ptr %[[VAR13]]
// CHECK: %[[VAR14:[A-Za-z0-9.]+]] = getelementptr inbounds { double, double }, ptr %[[VAR11]], i32 0, i32 0
// CHECK: %[[VAR15:[A-Za-z0-9.]+]] = load double, ptr %[[VAR14]], align 8
// CHECK: %[[VAR16:[A-Za-z0-9.]+]] = getelementptr inbounds { double, double }, ptr %[[VAR11]], i32 0, i32 1
// CHECK: %[[VAR17:[A-Za-z0-9.]+]] = load double, ptr %[[VAR16]], align 8
// CHECK: %{{[A-Za-z0-9.]+}} = call double @foo_double(double noundef %[[VAR15]], double noundef %[[VAR17]])

void bar_long_double(void) {
  foo_long_double(2.0L - 2.5Li);
}

// CHECK: define{{.*}} void @bar_long_double() [[NUW]] {
// CHECK: %[[VAR21:[A-Za-z0-9.]+]] = alloca { ppc_fp128, ppc_fp128 }, align 16
// CHECK: %[[VAR22:[A-Za-z0-9.]+]] = getelementptr inbounds { ppc_fp128, ppc_fp128 }, ptr %[[VAR21]], i32 0, i32 0
// CHECK: %[[VAR23:[A-Za-z0-9.]+]] = getelementptr inbounds { ppc_fp128, ppc_fp128 }, ptr %[[VAR21]], i32 0, i32 1
// CHECK: store ppc_fp128 0xM40000000000000000000000000000000, ptr %[[VAR22]]
// CHECK: store ppc_fp128 0xMC0040000000000008000000000000000, ptr %[[VAR23]]
// CHECK: %[[VAR24:[A-Za-z0-9.]+]] = getelementptr inbounds { ppc_fp128, ppc_fp128 }, ptr %[[VAR21]], i32 0, i32 0
// CHECK: %[[VAR25:[A-Za-z0-9.]+]] = load ppc_fp128, ptr %[[VAR24]], align 16
// CHECK: %[[VAR26:[A-Za-z0-9.]+]] = getelementptr inbounds { ppc_fp128, ppc_fp128 }, ptr %[[VAR21]], i32 0, i32 1
// CHECK: %[[VAR27:[A-Za-z0-9.]+]] = load ppc_fp128, ptr %[[VAR26]], align 16
// CHECK: %{{[A-Za-z0-9.]+}} = call ppc_fp128 @foo_long_double(ppc_fp128 noundef %[[VAR25]], ppc_fp128 noundef %[[VAR27]])

void bar_ibm128(void) {
  foo_ibm128(2.0L - 2.5Li);
}

// CHECK: define{{.*}} void @bar_ibm128() [[NUW]] {
// CHECK: %[[VAR21:[A-Za-z0-9.]+]] = alloca { ppc_fp128, ppc_fp128 }, align 16
// CHECK: %[[VAR22:[A-Za-z0-9.]+]] = getelementptr inbounds { ppc_fp128, ppc_fp128 }, ptr %[[VAR21]], i32 0, i32 0
// CHECK: %[[VAR23:[A-Za-z0-9.]+]] = getelementptr inbounds { ppc_fp128, ppc_fp128 }, ptr %[[VAR21]], i32 0, i32 1
// CHECK: store ppc_fp128 0xM40000000000000000000000000000000, ptr %[[VAR22]]
// CHECK: store ppc_fp128 0xMC0040000000000008000000000000000, ptr %[[VAR23]]
// CHECK: %[[VAR24:[A-Za-z0-9.]+]] = getelementptr inbounds { ppc_fp128, ppc_fp128 }, ptr %[[VAR21]], i32 0, i32 0
// CHECK: %[[VAR25:[A-Za-z0-9.]+]] = load ppc_fp128, ptr %[[VAR24]], align 16
// CHECK: %[[VAR26:[A-Za-z0-9.]+]] = getelementptr inbounds { ppc_fp128, ppc_fp128 }, ptr %[[VAR21]], i32 0, i32 1
// CHECK: %[[VAR27:[A-Za-z0-9.]+]] = load ppc_fp128, ptr %[[VAR26]], align 16
// CHECK: %{{[A-Za-z0-9.]+}} = call ppc_fp128 @foo_ibm128(ppc_fp128 noundef %[[VAR25]], ppc_fp128 noundef %[[VAR27]])

void bar_int(void) {
  foo_int(2 - 3i);
}

// CHECK: define{{.*}} void @bar_int() [[NUW]] {
// CHECK: %[[VAR31:[A-Za-z0-9.]+]] = alloca { i32, i32 }, align 4
// CHECK: %[[VAR32:[A-Za-z0-9.]+]] = getelementptr inbounds { i32, i32 }, ptr %[[VAR31]], i32 0, i32 0
// CHECK: %[[VAR33:[A-Za-z0-9.]+]] = getelementptr inbounds { i32, i32 }, ptr %[[VAR31]], i32 0, i32 1
// CHECK: store i32 2, ptr %[[VAR32]]
// CHECK: store i32 -3, ptr %[[VAR33]]
// CHECK: %[[VAR34:[A-Za-z0-9.]+]] = getelementptr inbounds { i32, i32 }, ptr %[[VAR31]], i32 0, i32 0
// CHECK: %[[VAR35:[A-Za-z0-9.]+]] = load i32, ptr %[[VAR34]], align 4
// CHECK: %[[VAR36:[A-Za-z0-9.]+]] = getelementptr inbounds { i32, i32 }, ptr %[[VAR31]], i32 0, i32 1
// CHECK: %[[VAR37:[A-Za-z0-9.]+]] = load i32, ptr %[[VAR36]], align 4
// CHECK: %{{[A-Za-z0-9.]+}} = call signext i32 @foo_int(i32 noundef %[[VAR35]], i32 noundef %[[VAR37]])

void bar_short(void) {
  foo_short(2 - 3i);
}

// CHECK: define{{.*}} void @bar_short() [[NUW]] {
// CHECK: %[[VAR41:[A-Za-z0-9.]+]] = alloca { i16, i16 }, align 2
// CHECK: %[[VAR42:[A-Za-z0-9.]+]] = getelementptr inbounds { i16, i16 }, ptr %[[VAR41]], i32 0, i32 0
// CHECK: %[[VAR43:[A-Za-z0-9.]+]] = getelementptr inbounds { i16, i16 }, ptr %[[VAR41]], i32 0, i32 1
// CHECK: store i16 2, ptr %[[VAR42]]
// CHECK: store i16 -3, ptr %[[VAR43]]
// CHECK: %[[VAR44:[A-Za-z0-9.]+]] = getelementptr inbounds { i16, i16 }, ptr %[[VAR41]], i32 0, i32 0
// CHECK: %[[VAR45:[A-Za-z0-9.]+]] = load i16, ptr %[[VAR44]], align 2
// CHECK: %[[VAR46:[A-Za-z0-9.]+]] = getelementptr inbounds { i16, i16 }, ptr %[[VAR41]], i32 0, i32 1
// CHECK: %[[VAR47:[A-Za-z0-9.]+]] = load i16, ptr %[[VAR46]], align 2
// CHECK: %{{[A-Za-z0-9.]+}} = call signext i16 @foo_short(i16 noundef %[[VAR45]], i16 noundef %[[VAR47]])

void bar_char(void) {
  foo_char(2 - 3i);
}

// CHECK: define{{.*}} void @bar_char() [[NUW]] {
// CHECK: %[[VAR51:[A-Za-z0-9.]+]] = alloca { i8, i8 }, align 1
// CHECK: %[[VAR52:[A-Za-z0-9.]+]] = getelementptr inbounds { i8, i8 }, ptr %[[VAR51]], i32 0, i32 0
// CHECK: %[[VAR53:[A-Za-z0-9.]+]] = getelementptr inbounds { i8, i8 }, ptr %[[VAR51]], i32 0, i32 1
// CHECK: store i8 2, ptr %[[VAR52]]
// CHECK: store i8 -3, ptr %[[VAR53]]
// CHECK: %[[VAR54:[A-Za-z0-9.]+]] = getelementptr inbounds { i8, i8 }, ptr %[[VAR51]], i32 0, i32 0
// CHECK: %[[VAR55:[A-Za-z0-9.]+]] = load i8, ptr %[[VAR54]], align 1
// CHECK: %[[VAR56:[A-Za-z0-9.]+]] = getelementptr inbounds { i8, i8 }, ptr %[[VAR51]], i32 0, i32 1
// CHECK: %[[VAR57:[A-Za-z0-9.]+]] = load i8, ptr %[[VAR56]], align 1
// CHECK: %{{[A-Za-z0-9.]+}} = call signext i8 @foo_char(i8 noundef %[[VAR55]], i8 noundef %[[VAR57]])

void bar_long(void) {
  foo_long(2L - 3Li);
}

// CHECK: define{{.*}} void @bar_long() [[NUW]] {
// CHECK: %[[VAR61:[A-Za-z0-9.]+]] = alloca { i64, i64 }, align 8
// CHECK: %[[VAR62:[A-Za-z0-9.]+]] = getelementptr inbounds { i64, i64 }, ptr %[[VAR61]], i32 0, i32 0
// CHECK: %[[VAR63:[A-Za-z0-9.]+]] = getelementptr inbounds { i64, i64 }, ptr %[[VAR61]], i32 0, i32 1
// CHECK: store i64 2, ptr %[[VAR62]]
// CHECK: store i64 -3, ptr %[[VAR63]]
// CHECK: %[[VAR64:[A-Za-z0-9.]+]] = getelementptr inbounds { i64, i64 }, ptr %[[VAR61]], i32 0, i32 0
// CHECK: %[[VAR65:[A-Za-z0-9.]+]] = load i64, ptr %[[VAR64]], align 8
// CHECK: %[[VAR66:[A-Za-z0-9.]+]] = getelementptr inbounds { i64, i64 }, ptr %[[VAR61]], i32 0, i32 1
// CHECK: %[[VAR67:[A-Za-z0-9.]+]] = load i64, ptr %[[VAR66]], align 8
// CHECK: %{{[A-Za-z0-9.]+}} = call i64 @foo_long(i64 noundef %[[VAR65]], i64 noundef %[[VAR67]])

void bar_long_long(void) {
  foo_long_long(2LL - 3LLi);
}

// CHECK: define{{.*}} void @bar_long_long() [[NUW]] {
// CHECK: %[[VAR71:[A-Za-z0-9.]+]] = alloca { i64, i64 }, align 8
// CHECK: %[[VAR72:[A-Za-z0-9.]+]] = getelementptr inbounds { i64, i64 }, ptr %[[VAR71]], i32 0, i32 0
// CHECK: %[[VAR73:[A-Za-z0-9.]+]] = getelementptr inbounds { i64, i64 }, ptr %[[VAR71]], i32 0, i32 1
// CHECK: store i64 2, ptr %[[VAR72]]
// CHECK: store i64 -3, ptr %[[VAR73]]
// CHECK: %[[VAR74:[A-Za-z0-9.]+]] = getelementptr inbounds { i64, i64 }, ptr %[[VAR71]], i32 0, i32 0
// CHECK: %[[VAR75:[A-Za-z0-9.]+]] = load i64, ptr %[[VAR74]], align 8
// CHECK: %[[VAR76:[A-Za-z0-9.]+]] = getelementptr inbounds { i64, i64 }, ptr %[[VAR71]], i32 0, i32 1
// CHECK: %[[VAR77:[A-Za-z0-9.]+]] = load i64, ptr %[[VAR76]], align 8
// CHECK: %{{[A-Za-z0-9.]+}} = call i64 @foo_long_long(i64 noundef %[[VAR75]], i64 noundef %[[VAR77]])

// CHECK: attributes [[NUW]] = { noinline nounwind{{.*}} }
