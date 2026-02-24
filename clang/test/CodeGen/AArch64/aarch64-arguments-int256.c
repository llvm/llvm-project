// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Verify AArch64 IR generation for __int256_t arguments and returns.

// CHECK-LABEL: define{{.*}} i256 @f_ret256(i256 noundef %a)
__int256_t f_ret256(__int256_t a) { return a; }

// CHECK-LABEL: define{{.*}} i256 @f_ret256u(i256 noundef %a)
__uint256_t f_ret256u(__uint256_t a) { return a; }

// Multiple 256-bit args
// CHECK-LABEL: define{{.*}} i256 @f_two256(i256 noundef %a, i256 noundef %b)
__int256_t f_two256(__int256_t a, __int256_t b) { return a + b; }

// Mixed argument sizes: 256-bit with smaller types
// CHECK-LABEL: define{{.*}} i256 @f_mixed(i64 noundef %x, i256 noundef %a, i32 noundef %y)
__int256_t f_mixed(long long x, __int256_t a, int y) { return a; }

// Register exhaustion: 3 i256 args still passed directly
// CHECK-LABEL: define{{.*}} i256 @f_three256(i256 noundef %a, i256 noundef %b, i256 noundef %c)
__int256_t f_three256(__int256_t a, __int256_t b, __int256_t c) { return a + b + c; }

// Struct containing a 256-bit integer: passed/returned via sret/indirect
struct s256 { __int256_t val; };

// CHECK-LABEL: define{{.*}} void @f_struct256(ptr dead_on_unwind noalias writable sret(%struct.s256) align 16 %{{.*}}, ptr noundef dead_on_return %s)
struct s256 f_struct256(struct s256 s) { return s; }

// Nested struct with __int256: also indirect
struct nested256 { int x; __int256_t val; int y; };

// CHECK-LABEL: define{{.*}} void @f_nested256(ptr dead_on_unwind noalias writable sret(%struct.nested256) align 16 %{{.*}}, ptr noundef dead_on_return %s)
struct nested256 f_nested256(struct nested256 s) { return s; }

// Packed struct with __int256
struct __attribute__((packed)) packed256 { char c; __int256_t val; };

// CHECK-LABEL: define{{.*}} void @f_packed256(ptr dead_on_unwind noalias writable sret(%struct.packed256) align 1 %{{.*}}, ptr noundef dead_on_return %s)
struct packed256 f_packed256(struct packed256 s) { return s; }
