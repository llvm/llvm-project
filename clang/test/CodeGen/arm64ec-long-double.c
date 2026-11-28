// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LD64
// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-80 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LD80
// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-128 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LD128

// -mlong-double-80 gives 'long double' the layout of x86_64-w64-mingw32 so that
// data can be shared with x64 code, but values cannot be computed with.

int ld_size = sizeof(long double);
int ld_align = _Alignof(long double);
// LD64:  @ld_size = dso_local global i32 8
// LD64:  @ld_align = dso_local global i32 8
// LD80:  @ld_size = dso_local global i32 16
// LD80:  @ld_align = dso_local global i32 16
// LD128: @ld_size = dso_local global i32 16
// LD128: @ld_align = dso_local global i32 16

// LD64:  @g = dso_local global double 1.500000e+00, align 8
// LD80:  @g = dso_local global x86_fp80 1.500000e+00, align 16
// LD128: @g = dso_local global fp128 1.500000e+00, align 16
long double g = 1.5L;

struct S {
  long double x;
};
// LD64:  @gs = dso_local global %struct.S { double 2.500000e+00 }, align 8
// LD80:  @gs = dso_local global %struct.S { x86_fp80 2.500000e+00 }, align 16
// LD128: @gs = dso_local global %struct.S { fp128 2.500000e+00 }, align 16
struct S gs = {2.5L};

// The x87 format is not a homogeneous floating-point aggregate base type.
// LD64:  define {{.*}}void @pass_struct([1 x double] %s.coerce)
// LD80:  define {{.*}}void @pass_struct(i128 %s.coerce)
// LD128: define {{.*}}void @pass_struct([1 x fp128] %s.coerce)
void pass_struct(struct S s) {}

// CHECK-LABEL: define {{.*}}void @copy(
// LD80:          call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{.*}}, ptr align 16 %{{.*}}, i64 16, i1 false)
void copy(struct S *d, struct S *s) { *d = *s; }

// CHECK-LABEL: define {{.*}}ptr @addr(
long double *addr(void) { return &g; }
