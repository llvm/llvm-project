// RUN: %clang_cc1 -triple powerpc64le-linux-gnu -emit-llvm -o - %s \
// RUN:   -mabi=ieeelongdouble | FileCheck --check-prefix=IEEE128 %s
// RUN: %clang_cc1 -triple powerpc64le-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck --check-prefix=PPC128 %s

long double x;
char buf[20];

// IEEE128-LABEL: define dso_local void @test_printf
// IEEE128: call signext i32 (ptr, ...) @__printfieee128
// PPC128-LABEL: define dso_local void @test_printf
// PPC128: call signext i32 (ptr, ...) @printf
void test_printf(void) {
  __builtin_printf("%.Lf", x);
}

// IEEE128-LABEL: define dso_local void @test_vsnprintf
// IEEE128: call signext i32 @__vsnprintfieee128
// PPC128-LABEL: define dso_local void @test_vsnprintf
// PPC128: call signext i32 @vsnprintf
void test_vsnprintf(int n, ...) {
  __builtin_va_list va;
  __builtin_va_start(va, n);
  __builtin_vsnprintf(buf, 20, "%.Lf", va);
  __builtin_va_end(va);
}

// IEEE128-LABEL: define dso_local void @test_vsprintf
// IEEE128: call signext i32 @__vsprintfieee128
// PPC128-LABEL: define dso_local void @test_vsprintf
// PPC128: call signext i32 @vsprintf
void test_vsprintf(int n, ...) {
  __builtin_va_list va;
  __builtin_va_start(va, n);
  __builtin_vsprintf(buf, "%.Lf", va);
  __builtin_va_end(va);
}

// IEEE128-LABEL: define dso_local void @test_sprintf
// IEEE128: call signext i32 (ptr, ptr, ...) @__sprintfieee128
// PPC128-LABEL: define dso_local void @test_sprintf
// PPC128: call signext i32 (ptr, ptr, ...) @sprintf
void test_sprintf(void) {
  __builtin_sprintf(buf, "%.Lf", x);
}

// IEEE128-LABEL: define dso_local void @test_snprintf
// IEEE128: call signext i32 (ptr, i64, ptr, ...) @__snprintfieee128
// PPC128-LABEL: define dso_local void @test_snprintf
// PPC128: call signext i32 (ptr, i64, ptr, ...) @snprintf
void test_snprintf(void) {
  __builtin_snprintf(buf, 20, "%.Lf", x);
}

// IEEE128-LABEL: define dso_local void @test_scanf
// IEEE128: call signext i32 (ptr, ...) @__scanfieee128
// PPC128-LABEL: define dso_local void @test_scanf
// PPC128: call signext i32 (ptr, ...) @scanf
void test_scanf(int *x) {
  __builtin_scanf("%d", x);
}

// IEEE128-LABEL: define dso_local void @test_sscanf
// IEEE128: call signext i32 (ptr, ptr, ...) @__sscanfieee128
// PPC128-LABEL: define dso_local void @test_sscanf
// PPC128: call signext i32 (ptr, ptr, ...) @sscanf
void test_sscanf(int *x) {
  __builtin_sscanf(buf, "%d", x);
}

// IEEE128-LABEL: define dso_local void @test_vprintf
// IEEE128: call signext i32 @__vprintfieee128
// PPC128-LABEL: define dso_local void @test_vprintf
// PPC128: call signext i32 @vprintf
void test_vprintf(const char *fmt, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, fmt);
  __builtin_vprintf(fmt, args);
  __builtin_va_end(args);
}

// IEEE128-LABEL: define dso_local void @test_vscanf
// IEEE128: call signext i32 @__vscanfieee128
// PPC128-LABEL: define dso_local void @test_vscanf
// PPC128: call signext i32 @vscanf
void test_vscanf(const char *fmt, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, fmt);
  __builtin_vscanf(fmt, args);
  __builtin_va_end(args);
}

// IEEE128-LABEL: define dso_local void @test_vsscanf
// IEEE128: call signext i32 @__vsscanfieee128
// PPC128-LABEL: define dso_local void @test_vsscanf
// PPC128: call signext i32 @vsscanf
void test_vsscanf(const char *fmt, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, fmt);
  __builtin_vsscanf(buf, fmt, args);
  __builtin_va_end(args);
}

// IEEE128-LABEL: define dso_local void @test_snprintf_chk
// IEEE128: call signext i32 (ptr, i64, i32, i64, ptr, ...) @__snprintf_chkieee128
// PPC128-LABEL: define dso_local void @test_snprintf_chk
// PPC128: call signext i32 (ptr, i64, i32, i64, ptr, ...) @__snprintf_chk
void test_snprintf_chk(long double x) {
  __builtin___snprintf_chk(buf, 20, 1, 20, "%.Lf", x);
}

// GLIBC has special handling of 'nexttoward'

// IEEE128-LABEL: define dso_local fp128 @test_nexttoward
// IEEE128: call fp128 @__nexttowardieee128
// PPC128-LABEL: define dso_local ppc_fp128 @test_nexttoward
// PPC128: call ppc_fp128 @nexttowardl
long double test_nexttoward(long double a, long double b) {
  return __builtin_nexttowardl(a, b);
}
