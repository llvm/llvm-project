// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Verify AArch64 handles many __int256 arguments (register exhaustion).
// Each __int256 consumes 4 GPRs (x0-x3, x4-x7), so the 3rd+ arg must
// spill to the stack when the backend lowers this.

// CHECK-LABEL: define{{.*}} i256 @f_five(i256 noundef %a, i256 noundef %b, i256 noundef %c, i256 noundef %d, i256 noundef %e)
// CHECK: add nsw i256
__int256 f_five(__int256 a, __int256 b, __int256 c, __int256 d, __int256 e) {
  return a + b + c + d + e;
}

// Mixed argument sizes: smaller args consume individual GPRs, then __int256
// takes 4 GPRs each.
// CHECK-LABEL: define{{.*}} i256 @f_mixed(i32 noundef %x, i256 noundef %a, i64 noundef %y, i256 noundef %b, i32 noundef %z)
// CHECK: add nsw i256
__int256 f_mixed(int x, __int256 a, long long y, __int256 b, int z) {
  return a + b;
}

// Struct containing __int256: must go indirect per AAPCS (>16 bytes)
struct s256 { __int256 val; };

// CHECK-LABEL: define{{.*}} void @f_struct(ptr{{.*}}sret(%struct.s256) align 16 %{{.*}}, ptr noundef dead_on_return %s)
struct s256 f_struct(struct s256 s) { return s; }

// Verify direct scalar __int256 return (even though struct s256 is indirect)
// CHECK-LABEL: define{{.*}} i256 @f_scalar_ret(i256 noundef %x)
// CHECK: ret i256
__int256 f_scalar_ret(__int256 x) { return x; }
