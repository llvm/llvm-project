// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Verify IR generation for __int256_t compound assignment and increment ops.
// On x86-64, __int256 value params use byval and returns use sret.

// CHECK-LABEL: define{{.*}} void @test_add_assign(ptr noundef %p, ptr noundef byval(i256) align 16 %0)
// CHECK: add nsw i256
void test_add_assign(__int256_t *p, __int256_t v) { *p += v; }

// CHECK-LABEL: define{{.*}} void @test_sub_assign(ptr noundef %p, ptr noundef byval(i256) align 16 %0)
// CHECK: sub nsw i256
void test_sub_assign(__int256_t *p, __int256_t v) { *p -= v; }

// CHECK-LABEL: define{{.*}} void @test_mul_assign(ptr noundef %p, ptr noundef byval(i256) align 16 %0)
// CHECK: mul nsw i256
void test_mul_assign(__int256_t *p, __int256_t v) { *p *= v; }

// CHECK-LABEL: define{{.*}} void @test_shl_assign(ptr noundef %p, i32 noundef %n)
// CHECK: shl i256
void test_shl_assign(__int256_t *p, int n) { *p <<= n; }

// CHECK-LABEL: define{{.*}} void @test_shr_assign(ptr noundef %p, i32 noundef %n)
// CHECK: ashr i256
void test_shr_assign(__int256_t *p, int n) { *p >>= n; }

// CHECK-LABEL: define{{.*}} void @test_ushr_assign(ptr noundef %p, i32 noundef %n)
// CHECK: lshr i256
void test_ushr_assign(__uint256_t *p, int n) { *p >>= n; }

// CHECK-LABEL: define{{.*}} void @test_and_assign(ptr noundef %p, ptr noundef byval(i256) align 16 %0)
// CHECK: and i256
void test_and_assign(__int256_t *p, __int256_t v) { *p &= v; }

// CHECK-LABEL: define{{.*}} void @test_or_assign(ptr noundef %p, ptr noundef byval(i256) align 16 %0)
// CHECK: or i256
void test_or_assign(__int256_t *p, __int256_t v) { *p |= v; }

// CHECK-LABEL: define{{.*}} void @test_xor_assign(ptr noundef %p, ptr noundef byval(i256) align 16 %0)
// CHECK: xor i256
void test_xor_assign(__int256_t *p, __int256_t v) { *p ^= v; }

// CHECK-LABEL: define{{.*}} void @test_pre_inc(ptr{{.*}}sret(i256){{.*}}, ptr noundef byval(i256) align 16 %0)
// CHECK: add nsw i256 %{{.*}}, 1
__int256_t test_pre_inc(__int256_t a) { return ++a; }

// CHECK-LABEL: define{{.*}} void @test_pre_dec(ptr{{.*}}sret(i256){{.*}}, ptr noundef byval(i256) align 16 %0)
// CHECK: add nsw i256 %{{.*}}, -1
__int256_t test_pre_dec(__int256_t a) { return --a; }

// CHECK-LABEL: define{{.*}} void @test_post_inc(ptr{{.*}}sret(i256){{.*}}, ptr noundef %p)
// CHECK: add nsw i256 %{{.*}}, 1
__int256_t test_post_inc(__int256_t *p) { return (*p)++; }

// CHECK-LABEL: define{{.*}} void @test_post_dec(ptr{{.*}}sret(i256){{.*}}, ptr noundef %p)
// CHECK: add nsw i256 %{{.*}}, -1
__int256_t test_post_dec(__int256_t *p) { return (*p)--; }
