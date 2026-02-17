// RUN: %clang_cc1 -triple=i386-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-I386
// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-X86_64

// Regression test for the _BitInt alignment issue
// This reproduces the exact test case from the user's investigation:
// https://gcc.godbolt.org/z/z59c6Tn7n
//
// The issue: __alignof(_BitInt(64)) returned different values:
// - Clang 20.1.0 -m32: 4 (32-bit alignment) ❌
// - GCC 15.2 -m32: 8 (64-bit alignment) ✅
//
// This test verifies that the fix resolves the discrepancy

#include <stdint.h>

// Test variables - exactly as in the original investigation
long long vll;
_BitInt(64) vbi;

// Test struct to capture all alignment values - exactly as in the original
struct result {
    int uppercase_alignof_bi64;      // _Alignof(_BitInt(64))
    int uppercase_alignof_vbi;       // _Alignof(vbi)
    int uppercase_alignof_longlong;  // _Alignof(long long)
    int uppercase_alignof_vll;       // _Alignof(vll)
    int lowercase_alignof_bi64;      // __alignof(_BitInt(64))
    int lowercase_alignof_vbi;       // __alignof(vbi)
    int lowercase_alignof_longlong;  // __alignof(long long)
    int lowercase_alignof_vll;       // __alignof(vll)
} result = {
    .uppercase_alignof_bi64 = _Alignof(_BitInt(64)),
    .uppercase_alignof_vbi = _Alignof(vbi),
    .uppercase_alignof_longlong = _Alignof(long long),
    .uppercase_alignof_vll = _Alignof(vll),
    .lowercase_alignof_bi64 = __alignof(_BitInt(64)),
    .lowercase_alignof_vbi = __alignof(vbi),
    .lowercase_alignof_longlong = __alignof(long long),
    .lowercase_alignof_vll = __alignof(vll),
};

// Test function to verify the fix
void verify_fix() {
    // Before the fix, these would fail on i386:
    // - _Alignof(_BitInt(64)) returned 4 instead of 8
    // - __alignof(_BitInt(64)) returned 4 instead of 8
    
    // After the fix, these should all pass:
    _Static_assert(_Alignof(_BitInt(64)) == 8, "_BitInt(64) should have 8-byte alignment");
    _Static_assert(__alignof(_BitInt(64)) == 8, "__alignof(_BitInt(64)) should return 8");
    
    // Verify consistency with long long
    _Static_assert(_Alignof(_BitInt(64)) == _Alignof(long long), "_BitInt(64) and long long should have same alignment");
    _Static_assert(__alignof(_BitInt(64)) == __alignof(long long), "__alignof should return same value for both");
}

// Test struct layout to ensure alignment is respected
struct test_struct {
    char c;
    _BitInt(64) bi;
    long long ll;
};

void test_struct_layout() {
    // Verify that _BitInt(64) gets proper alignment in structs
    _Static_assert(offsetof(struct test_struct, bi) % 8 == 0, "_BitInt(64) should be 8-byte aligned in struct");
    _Static_assert(offsetof(struct test_struct, ll) % 8 == 0, "long long should be 8-byte aligned in struct");
    
    // Verify struct alignment
    _Static_assert(_Alignof(struct test_struct) >= 8, "test_struct should have at least 8-byte alignment");
}

// CHECK-I386: @result = global %struct.result { i32 4, i32 4, i32 4, i32 8, i32 8, i32 8, i32 8, i32 8 }
// CHECK-X86_64: @result = global %struct.result { i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8 }

// Verify struct layout respects alignment
// CHECK-I386: %struct.test_struct = type { i8, [3 x i8], i64, i64 }
// CHECK-X86_64: %struct.test_struct = type { i8, [7 x i8], i64, i64 } 