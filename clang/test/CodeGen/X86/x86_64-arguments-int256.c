// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Verify X86-64 IR generation for __int256_t arguments and returns.
// Per the SysV ABI, types exceeding two eightbytes (128 bits) are passed
// and returned in memory (sret/byval).

// CHECK-LABEL: define{{.*}} void @f_ret256(ptr dead_on_unwind noalias writable sret(i256) align 16 %{{.*}}, ptr noundef byval(i256) align 16 %0)
__int256_t f_ret256(__int256_t a) { return a; }

// CHECK-LABEL: define{{.*}} void @f_ret256u(ptr dead_on_unwind noalias writable sret(i256) align 16 %{{.*}}, ptr noundef byval(i256) align 16 %0)
__uint256_t f_ret256u(__uint256_t a) { return a; }

// Multiple 256-bit args
// CHECK-LABEL: define{{.*}} void @f_two256(ptr dead_on_unwind noalias writable sret(i256) align 16 %{{.*}}, ptr noundef byval(i256) align 16 %0, ptr noundef byval(i256) align 16 %1)
__int256_t f_two256(__int256_t a, __int256_t b) { return a + b; }

// Mixed argument sizes: 256-bit with smaller types
// CHECK-LABEL: define{{.*}} void @f_mixed(ptr dead_on_unwind noalias writable sret(i256) align 16 %{{.*}}, i64 noundef %x, ptr noundef byval(i256) align 16 %0, i32 noundef %y)
__int256_t f_mixed(long long x, __int256_t a, int y) { return a; }

// 128-bit: still returned directly in registers (2 eightbytes)
// CHECK-LABEL: define{{.*}} i128 @f_ret128(i128 noundef %a)
__int128_t f_ret128(__int128_t a) { return a; }

// 3 i256 args: all passed via byval pointers
// CHECK-LABEL: define{{.*}} void @f_three256(ptr dead_on_unwind noalias writable sret(i256) align 16 %{{.*}}, ptr noundef byval(i256) align 16 %0, ptr noundef byval(i256) align 16 %1, ptr noundef byval(i256) align 16 %2)
__int256_t f_three256(__int256_t a, __int256_t b, __int256_t c) { return a + b + c; }

// Struct containing a 256-bit integer: passed/returned via sret/byval
struct s256 { __int256_t val; };

// CHECK-LABEL: define{{.*}} void @f_struct256(ptr dead_on_unwind noalias writable sret(%struct.s256) align 16 %{{.*}}, ptr noundef byval(%struct.s256) align 16 %s)
struct s256 f_struct256(struct s256 s) { return s; }

// Nested struct with __int256
struct nested256 { int x; __int256_t val; int y; };

// CHECK-LABEL: define{{.*}} void @f_nested256(ptr dead_on_unwind noalias writable sret(%struct.nested256) align 16 %{{.*}}, ptr noundef byval(%struct.nested256) align 16 %s)
struct nested256 f_nested256(struct nested256 s) { return s; }
