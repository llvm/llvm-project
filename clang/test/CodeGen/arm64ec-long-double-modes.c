// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-64 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=LD64
// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-80 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=LD80
// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-128 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=LD128

// The three widths select the format of `long double`; only the 80-bit one is
// the software x87 format that goes through the ARM64EC runtime helpers.

long double g = 1.0L;
long double add(long double a, long double b) { return a + b; }
int cmp(long double a, long double b) { return a < b; }

// LD64: @g = {{.*}}global double 1.000000e+00
// LD64-LABEL: define {{.*}}double @add(double noundef %a, double noundef %b)
// LD64: fadd double
// LD64-LABEL: define {{.*}}i32 @cmp(double noundef %a, double noundef %b)
// LD64: fcmp olt double
// LD64-NOT: __arm64ec_

// The 80-bit format stays a real `x86_fp80` in the IR so that the middle-end
// can optimize it; the ABI passes and returns it as a 16-byte image, and the
// backend softens every operation to a compiler-rt helper.  The padding after
// the 80 bits of the slot is not represented by the IR type.
// LD80: @g = {{.*}}global x86_fp80 1.000000e+00, align 16
// LD80-LABEL: define {{.*}}i128 @add(i128 {{.*}}%a.coerce, i128 {{.*}}%b.coerce)
// LD80: fadd x86_fp80
// LD80-NOT: __arm64ec_
// LD80-LABEL: define {{.*}}i32 @cmp(i128 {{.*}}%a.coerce, i128 {{.*}}%b.coerce)
// LD80: fcmp olt x86_fp80
// LD80-NOT: __arm64ec_

// LD128: @g = {{.*}}global fp128
// LD128-LABEL: define {{.*}}fp128 @add(fp128 noundef %a, fp128 noundef %b)
// LD128-LABEL: define {{.*}}i32 @cmp(fp128 noundef %a, fp128 noundef %b)
// LD128-NOT: __arm64ec_
