// RUN: %clang_cc1 -triple riscv64 -target-feature +d -target-abi lp64d \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// preserve_none has more integer argument registers than the standard calling
// convention, but aggregates are still classified as if there were eight:
// whether a struct is passed in an integer and a floating-point register, or
// coerced to integers, doesn't depend on the calling convention.

struct ld { long l; double d; };
struct ll { long a; long b; };

// CHECK-LABEL: define{{.*}} preserve_nonecc i64 @ld_first(i64 %0, double %1)
__attribute__((preserve_none)) long ld_first(struct ld s) { return s.l; }

// CHECK-LABEL: define{{.*}} preserve_nonecc i64 @ld_after_eight(i64 noundef %a, i64 noundef %b, i64 noundef %c, i64 noundef %d, i64 noundef %e, i64 noundef %f, i64 noundef %g, i64 noundef %h, [2 x i64] %s.coerce)
__attribute__((preserve_none)) long ld_after_eight(long a, long b, long c, long d, long e, long f, long g, long h, struct ld s) { return s.l; }

// CHECK-LABEL: define{{.*}} preserve_nonecc i64 @ll_after_seven(i64 noundef %a, i64 noundef %b, i64 noundef %c, i64 noundef %d, i64 noundef %e, i64 noundef %f, i64 noundef %g, [2 x i64] %s.coerce)
__attribute__((preserve_none)) long ll_after_seven(long a, long b, long c, long d, long e, long f, long g, struct ll s) { return s.b; }
