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

// GLIBC has special handling of 'nexttoward'

// IEEE128-LABEL: define dso_local fp128 @test_nexttoward
// IEEE128: call fp128 @__nexttowardieee128
// PPC128-LABEL: define dso_local ppc_fp128 @test_nexttoward
// PPC128: call ppc_fp128 @nexttowardl
long double test_nexttoward(long double a, long double b) {
  return __builtin_nexttowardl(a, b);
}
